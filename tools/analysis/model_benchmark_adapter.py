import importlib
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings

from mmseg.models import build_segmentor


class BenchmarkAdapterRegistry:

    def __init__(self):
        self._modules = {}

    def register_module(self, name):

        def decorator(cls):
            self._modules[name] = cls
            return cls

        return decorator

    def build(self, name):
        if name not in self._modules:
            raise KeyError(f'Unknown benchmark adapter: {name}')
        return self._modules[name]()


BENCHMARK_ADAPTERS = BenchmarkAdapterRegistry()


def _import_custom_modules(cfg):
    custom_imports = cfg.get('custom_imports', None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)


def _strip_prefix(key):
    prefixes = ('module.', 'model.', 'student.', 'segmentor.')
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _extract_checkpoint_state(checkpoint):
    for field in ('state_dict', 'model', 'student', 'ema_model'):
        if isinstance(checkpoint, dict) and field in checkpoint:
            state = checkpoint[field]
            if isinstance(state, dict):
                return state, field
    if isinstance(checkpoint, dict):
        return checkpoint, 'raw'
    raise TypeError('Unsupported checkpoint object.')


def load_checkpoint_compatible(model, checkpoint_path, map_location='cpu'):
    if checkpoint_path is None:
        return dict(
            checkpoint_loaded=False,
            checkpoint_field=None,
            matched_keys=0,
            missing_keys=[],
            unexpected_keys=[],
            shape_mismatch_keys=[])

    import torch

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    raw_state, field = _extract_checkpoint_state(checkpoint)
    model_state = model.state_dict()
    cleaned = OrderedDict()
    unexpected = []
    shape_mismatch = []
    matched = 0

    for key, value in raw_state.items():
        clean_key = _strip_prefix(key)
        if clean_key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[clean_key].shape):
            shape_mismatch.append(key)
            continue
        cleaned[clean_key] = value
        matched += 1

    missing = [key for key in model_state.keys() if key not in cleaned]
    model.load_state_dict(cleaned, strict=False)
    return dict(
        checkpoint_loaded=True,
        checkpoint_field=field,
        matched_keys=matched,
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatch_keys=shape_mismatch)


def unwrap_deploy_model(model):
    original_type = type(model).__name__
    current = model
    visited = set()
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, 'module'):
            current = current.module
            continue
        if hasattr(current, 'get_model') and callable(current.get_model):
            current = current.get_model()
            continue
        for attr in ('student', 'segmentor'):
            if hasattr(current, attr):
                current = getattr(current, attr)
                break
        else:
            break
    return current, original_type, type(current).__name__


@BENCHMARK_ADAPTERS.register_module('default')
class DefaultBenchmarkAdapter:

    def build_deploy_model(self, config_path, checkpoint_path=None, device='cpu'):
        cfg = Config.fromfile(config_path)
        _import_custom_modules(cfg)
        # cfg.model already carries train_cfg/test_cfg in this project. Passing
        # them again through build_segmentor() triggers mmseg's duplicate-config
        # assertion, so the deployment model is built from cfg.model directly.
        model = build_segmentor(cfg.model, train_cfg=None, test_cfg=None)
        model, original_type, deployed_type = unwrap_deploy_model(model)
        checkpoint_info = load_checkpoint_compatible(
            model, checkpoint_path, map_location='cpu')
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, cfg, dict(
            original_wrapper_type=original_type,
            deployed_model_type=deployed_type,
            checkpoint_info=checkpoint_info)

    def forward_deploy_logits(self, model, dummy_input, img_metas):
        if hasattr(model, 'encode_decode'):
            return model.encode_decode(dummy_input, img_metas)
        if hasattr(model, 'inference'):
            return model.inference(dummy_input, img_metas, rescale=False)
        if hasattr(model, 'forward_dummy'):
            return model.forward_dummy(dummy_input)
        return model(dummy_input)


def build_adapter(name):
    if name not in BENCHMARK_ADAPTERS._modules:
        importlib.import_module(__name__)
    return BENCHMARK_ADAPTERS.build(name)
