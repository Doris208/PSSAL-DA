import json
import math
import os
import statistics
import time
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def torch_inference_context():
    if hasattr(torch, 'inference_mode'):
        with torch.inference_mode():
            yield
    else:
        with torch.no_grad():
            yield


def make_img_metas(height=256, width=256, channels=6):
    return [
        dict(
            filename='benchmark_dummy.tif',
            ori_filename='benchmark_dummy.tif',
            ori_shape=(height, width, channels),
            img_shape=(height, width, channels),
            pad_shape=(height, width, channels),
            scale_factor=1.0,
            flip=False,
            flip_direction=None)
    ]


def count_parameters(model):
    seen = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
    return total


def collect_active_parameters(model, forward_fn, dummy_input, img_metas):
    active_modules = []
    handles = []

    def hook(module, inputs, outputs):
        active_modules.append(module)

    for module in model.modules():
        handles.append(module.register_forward_hook(hook))
    try:
        with torch_inference_context():
            forward_fn(model, dummy_input, img_metas)
    finally:
        for handle in handles:
            handle.remove()

    seen = set()
    total = 0
    for module in active_modules:
        for parameter in module.parameters(recurse=False):
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
    return total, [type(module).__name__ for module in active_modules]


def _conv2d_flops(module, inputs, output):
    x = inputs[0]
    batch = x.shape[0]
    out_h, out_w = output.shape[-2:]
    kernel_h, kernel_w = module.kernel_size
    groups = module.groups
    in_channels = module.in_channels
    out_channels = module.out_channels
    filters_per_channel = out_channels // groups
    conv_per_position = kernel_h * kernel_w * in_channels * filters_per_channel
    active_elements = batch * out_h * out_w
    macs = conv_per_position * active_elements
    if module.bias is not None:
        macs += out_channels * active_elements
    return macs


def _linear_flops(module, inputs, output):
    x = inputs[0]
    batch = x.numel() // x.shape[-1]
    macs = batch * module.in_features * module.out_features
    if module.bias is not None:
        macs += batch * module.out_features
    return macs


def _bn_flops(module, inputs, output):
    return output.numel() * 2


def _activation_flops(module, inputs, output):
    return output.numel()


def count_flops_hook(model, forward_fn, dummy_input, img_metas):
    supported = {
        nn.Conv2d: _conv2d_flops,
        nn.Linear: _linear_flops,
        nn.BatchNorm2d: _bn_flops,
        nn.ReLU: _activation_flops,
        nn.ReLU6: _activation_flops,
        nn.LeakyReLU: _activation_flops,
        nn.GELU: _activation_flops,
        nn.Sigmoid: _activation_flops,
        nn.AdaptiveAvgPool2d: lambda m, i, o: o.numel(),
        nn.AvgPool2d: lambda m, i, o: o.numel(),
        nn.MaxPool2d: lambda m, i, o: o.numel(),
        nn.Upsample: lambda m, i, o: o.numel(),
    }
    macs = 0
    unsupported = {}
    handles = []

    def hook(module, inputs, output):
        nonlocal macs
        if len(list(module.children())) > 0:
            return
        for cls, fn in supported.items():
            if isinstance(module, cls):
                macs += int(fn(module, inputs, output))
                return
        if len(list(module.parameters(recurse=False))) > 0:
            name = type(module).__name__
            unsupported[name] = unsupported.get(name, 0) + 1

    for module in model.modules():
        handles.append(module.register_forward_hook(hook))
    try:
        with torch_inference_context():
            forward_fn(model, dummy_input, img_metas)
    finally:
        for handle in handles:
            handle.remove()

    gmacs = macs / 1e9
    return dict(
        gmacs=gmacs,
        gflops=2.0 * gmacs,
        flop_backend='local_hook_counter',
        flop_convention='1 MAC = 2 FLOPs',
        unsupported_ops=unsupported,
        flop_coverage_status='OK' if not unsupported else 'NEEDS_REVIEW')


def benchmark_fps(model,
                  forward_fn,
                  dummy_input,
                  img_metas,
                  warmup=50,
                  iterations=300,
                  repeat=5):
    if not dummy_input.is_cuda:
        return dict(
            fps_mean=None,
            fps_std=None,
            fps_min=None,
            fps_max=None,
            latency_mean_ms=None,
            latency_p50_ms=None,
            latency_p95_ms=None,
            peak_memory_mb=None,
            status='SKIPPED',
            error_message='FPS benchmark requires CUDA input.')

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    all_fps = []
    latencies = []
    torch.cuda.reset_peak_memory_stats(dummy_input.device)
    with torch_inference_context():
        for _ in range(warmup):
            forward_fn(model, dummy_input, img_metas)
        torch.cuda.synchronize(dummy_input.device)
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(iterations):
                forward_fn(model, dummy_input, img_metas)
            torch.cuda.synchronize(dummy_input.device)
            elapsed = time.perf_counter() - start
            fps = iterations * dummy_input.shape[0] / elapsed
            all_fps.append(fps)
            latencies.append(elapsed * 1000.0 / iterations)
    peak_memory_mb = torch.cuda.max_memory_allocated(dummy_input.device) / 1024**2
    sorted_lat = sorted(latencies)
    p95_idx = min(len(sorted_lat) - 1, math.ceil(0.95 * len(sorted_lat)) - 1)
    return dict(
        fps_mean=statistics.mean(all_fps),
        fps_std=statistics.pstdev(all_fps) if len(all_fps) > 1 else 0.0,
        fps_min=min(all_fps),
        fps_max=max(all_fps),
        latency_mean_ms=statistics.mean(latencies),
        latency_p50_ms=statistics.median(latencies),
        latency_p95_ms=sorted_lat[p95_idx],
        peak_memory_mb=peak_memory_mb,
        status='OK',
        error_message=None)


def get_environment_info(device):
    info = dict(
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        device=str(device))
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update(
            gpu_name=props.name,
            gpu_memory_mb=props.total_memory / 1024**2)
    else:
        info.update(gpu_name=None, gpu_memory_mb=None)
    try:
        import mmcv
        info['mmcv_version'] = mmcv.__version__
    except Exception:
        info['mmcv_version'] = None
    try:
        import mmseg
        info['mmseg_version'] = mmseg.__version__
    except Exception:
        info['mmseg_version'] = None
    return info


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
