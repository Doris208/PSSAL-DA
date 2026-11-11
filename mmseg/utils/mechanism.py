"""DORIS observation utilities. Training and fixed-probe data remain separate."""
from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
import traceback
import uuid

import numpy as np
import torch


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def readonly_probe(model):
    """Restore RNG, heterogeneous module modes and inference scratch attributes."""
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    class_crops = {type(module): deepcopy(type(module).last_train_crop_box)
                   for module, _ in modes if hasattr(type(module), 'last_train_crop_box')}
    scratch = [(module, name, deepcopy(getattr(module, name)))
               for module, _ in modes for name in ('hr_crop_box', 'debug_output')
               if hasattr(module, name)]
    try:
        model.eval()
        for module, name, _ in scratch:
            if name == 'hr_crop_box':
                module.hr_crop_box = None  # A probe covers the full image, not a training crop.
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        for module, name, value in scratch:
            setattr(module, name, value)
        for cls, value in class_crops.items():
            cls.last_train_crop_box = value
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def student_test_metadata(model, image, metas):
    """Describe the existing Student test interface used for a display prediction."""
    student = model.get_model()
    normalization = metas[0].get('img_norm_cfg')
    if normalization is not None:
        normalization = {key: value.tolist() if isinstance(value, (np.ndarray, np.generic)) else deepcopy(value)
                         for key, value in normalization.items()}
    return dict(model_identity='Student', prediction_field='struct_prediction',
        forward_call='get_model()(img=[image], img_metas=[metas], return_loss=False, rescale=True)',
        class_reduction='argmax(dim=1)', eval=True, rescale=True,
        test_cfg=deepcopy(getattr(student, 'test_cfg', None)),
        scales=deepcopy(getattr(student, 'scales', None)),
        hr_crop_size=deepcopy(getattr(student, 'crop_size', None)),
        hr_slide_inference=getattr(student, 'hr_slide_inference', None),
        fp16_enabled=bool(getattr(student, 'fp16_enabled', False)),
        input_shape=list(image.shape), input_dtype=str(image.dtype),
        input_normalization=normalization,
        flip=metas[0].get('flip', False), display_crop=[30, 30, 226, 226])


MODULES = ('phy', 'struct', 'proto', 'edge')
MEANS = ('w_conf', 'w_base', 'r_phy', 'w_t_phy', 'w_t_ps', 'w_pre_proto',
         'w_t_proto', 'w_final', 'r_struct', 'e_prob', 'e_img', 'r_pair',
         'r_edge', 'boundary_weight', 'delta_w_phy', 'delta_w_ps', 'delta_w_proto')
BRANCH_FIELDS = {
    'phy': ('phy_response_mean_valid', 'phy_weight_before_mean',
            'phy_weight_after_mean', 'phy_downweight_ratio'),
    'struct': ('struct_response_mean_valid', 'struct_weight_before_mean', 'struct_weight_after_mean'),
    'proto': ('proto_weight_before_mean', 'proto_weight_after_mean',
              'proto_target_selected_ratio', 'proto_target_update_ratio'),
    'edge': ('edge_response_mean_valid', 'edge_supervision_mean_valid')}


def module_flags(model):
    return {name: bool(getattr(model, 'enable_' + ('struct_edge' if name == 'edge' else name), False))
            for name in MODULES}


def module_states(model):
    flags = module_flags(model)
    states = {name: 'disabled' for name in MODULES}
    for name in ('phy', 'struct'):
        if flags[name]:
            start = (model.phy_warmup_iters if name == 'phy' else
                     model.struct_gate.struct_warmup_iters)
            ramp = (model.phy_rampup_iters if name == 'phy' else
                    model.struct_gate.struct_rampup_iters)
            states[name] = ('warmup' if model.local_iter <= start else
                            'ramping' if model.local_iter < start + ramp else 'active')
    if flags['phy'] and getattr(model, 'phy_mode', 'legacy_binary') == 'hierarchical_gv':
        from mmseg.models.uda.phy_reliability import phy_gate_strength
        rho = phy_gate_strength(model.local_iter, **model.phy_gate_params)
        states['phy'] = ('warmup' if rho == 0 else 'ramping' if rho < 1 else 'active')
    if flags['struct']:
        rho = model.struct_gate.gate_strength(model.local_iter)
        states['struct'] = 'warmup' if rho == 0 else 'ramping' if rho < 1 else 'active'
    if flags['proto']:
        gate = model.proto_gate
        ready = int(gate.valid_proto.sum())
        states['proto'] = ('warmup' if model.local_iter < gate.weight_start else
                           'cache_unready' if ready < max(1, gate.min_valid_classes_for_weight)
                           else 'active')
    if flags['edge']:
        states['edge'] = 'ramping' if model._struct_edge_warmup_ratio() < 1 else 'active'
    return states


def probe_arrays(model, image, vi, metas, spatial_valid):
    """Only evaluate enabled modules; never construct a disabled module's evidence."""
    flags = module_flags(model)
    with readonly_probe(model), torch.cuda.amp.autocast(enabled=image.is_cuda):
        logits = model.get_ema_model().generate_pseudo_label(image, metas)
        if logits.shape[-2:] != image.shape[-2:]:
            raise ValueError('Probe teacher logits must cover the full input pixel grid')
        probabilities = logits.softmax(1)
        confidence, label = probabilities.max(1)
        valid = model._doris_target_valid_mask(label, spatial_valid)[:, 0]
        ratio = confidence.ge(model.pseudo_threshold).sum().item() / label.numel()
        weight = torch.ones(label.shape, device=image.device) * ratio * valid
        maps = dict(w_conf=confidence, w_base=weight, pseudo_label=label, valid_mask=valid)
        if flags['phy']:
            response, weight, gate = model._apply_phy_gate(vi, label, weight, {}, valid_mask=valid)
            if isinstance(response, dict):
                top_two = probabilities.topk(2, dim=1).values
                maps.update({key: value[:, 0] for key, value in response.items()
                             if not key.startswith('logit_')})
                maps.update(attenuation_phy=response['a_phy'][:, 0],
                            teacher_label=label, teacher_confidence=confidence,
                            teacher_top2_margin=top_two[:, 0] - top_two[:, 1],
                            vi_ndvi=vi[:, 0], vi_ndre=vi[:, 1], vi_cire=vi[:, 2])
                response = response['r_veg']
            maps.update(r_phy=response[:, 0], w_t_phy=weight,
                        m_phy=model._class_mask(label, model.phy_apply_to_classes),
                        delta_w_phy=weight - maps['w_base'])
        if flags['struct']:
            maps['w_before_struct'] = weight
            weight, responses = model.struct_gate.apply(
                target_probs=probabilities, pseudo_label=label, pseudo_weight=weight,
                target_confidence=confidence, iter=model.local_iter,
                collect_debug=False, log_stats=False, update_priors=False)
            maps.update(r_conf=confidence, r_loc=responses['R_continuity'],
                r_adj=responses['R_adj'], r_prob=responses['R_sem'], r_struct=responses['R_struct'],
                adjacency_available=responses['adjacency_available'],
                semantic_available=responses['semantic_available'])
            for key in ('confidence_available', 'continuity_available', 'evidence_count',
                        'target_cross_neighbor_count', 'source_cross_prior_available'):
                if key in responses:
                    maps[key] = responses[key]
            maps['w_after_struct'] = weight
        if flags['phy'] or flags['struct']:
            maps.update(w_t_ps=weight, delta_w_ps=weight - maps['w_base'])
        if flags['proto']:
            maps['w_pre_proto'] = weight
            features = model.get_model().extract_feat(image)
            feature_shape = list(model.proto_gate._select_feat(features).shape[-2:])
            proto_weight, _ = model.proto_gate.compute_weight(
                features, label, out_size=label.shape[-2:], iter=model.local_iter, read_only=True)
            final = weight * proto_weight.detach()
            maps.update(w_t_proto=final, delta_w_proto=final - weight)
        else:
            final = weight
        maps['w_final'] = final
        if flags['edge']:
            responses = model.doris_edge_responses(image, label, probabilities)
            maps.update({key: responses[source][:, 0] for key, source in
                (('e_prob', 'E_prob'), ('e_img', 'E_img'), ('r_pair', 'R_pair'), ('r_edge', 'R_edge'))})
            reliability = ((confidence - model.struct_edge_cfg.get('confidence_floor', .60)) /
                           model.struct_edge_cfg.get('confidence_range', .35)).clamp(0, 1)
            maps['boundary_weight'] = (maps['r_edge'] * reliability * valid *
                model.struct_edge_loss_weight_tgt * model._struct_edge_warmup_ratio())
            maps['teacher_pseudo_boundary'] = model.mechanism_boundary(label)[:, 0] & valid
            maps['edge_supervision'] = maps['boundary_weight'] * maps['teacher_pseudo_boundary']
        result = {key: value[0].detach().float().cpu().numpy() for key, value in maps.items()}
        result['proto_active'] = np.array(module_states(model)['proto'] == 'active')
        result['rule_iter'] = np.array(model.local_iter)
        result['teacher_logits_shape'] = np.array(logits.shape[-2:])
        if flags['proto']:
            result['prototype_feature_shape'] = np.array(feature_shape)
        result['pseudo_label'] = result['pseudo_label'].astype(np.uint8)
        for key in ('valid_mask', 'm_phy', 'teacher_pseudo_boundary', 'adjacency_available', 'semantic_available',
                    'confidence_available', 'continuity_available', 'source_cross_prior_available'):
            if key in result:
                result[key] = result[key].astype(bool)
        return result


def training_record(model, confidence, label, w_t_ps, final, valid, structural,
                    edge_response, boundary_weight, w_base=None, w_t_phy=None,
                    r_phy=None, edge_components=None, log_vars=None):
    """Sample enabled modules from actual pre-ClassMix tensors, without a forward."""
    flags, states = module_flags(model), module_states(model)
    if isinstance(r_phy, dict):
        hphy = r_phy
        r_phy = r_phy['r_veg']
    else:
        hphy = None
    branch_mode = model.mechanism_cfg.get('output_mode', 'branches') == 'branches'
    valid = valid[:, 0] if valid.dim() == 4 else valid
    count = int(valid.sum())
    fields = dict(w_conf=confidence, w_final=final)
    if w_base is not None:
        fields['w_base'] = w_base
    if flags['phy']:
        fields.update(r_phy=r_phy, w_t_phy=w_t_phy)
        if w_base is not None:
            fields['delta_w_phy'] = w_t_phy - w_base
    if flags['phy'] or flags['struct']:
        fields['w_t_ps'] = w_t_ps
        if w_base is not None:
            fields['delta_w_ps'] = w_t_ps - w_base
    if flags['struct']:
        fields['r_struct'] = structural['R_struct']
    if flags['proto']:
        fields.update(w_pre_proto=w_t_ps, w_t_proto=final, delta_w_proto=final - w_t_ps)
    if flags['edge']:
        fields.update(r_edge=edge_response, boundary_weight=boundary_weight)
        for key, source in (('e_prob', 'E_prob'), ('e_img', 'E_img'), ('r_pair', 'R_pair')):
            if edge_components is not None:
                fields[key] = edge_components[source]
    sums = {}
    for name, value in (fields.items() if not branch_mode or model.mechanism_cfg.get('detailed_output') else []):
        if value is not None:
            if value.dim() == 4:
                value = value[:, 0]
            sums[name] = value.detach()[valid].float().sum().item()
    gate, proto = model.proto_gate, []
    if flags['proto']:
        for cls, name in enumerate(model.class_names):
            diag = gate.last_target_diagnostics
            prefix = 'class_{}.'.format(cls)
            row = dict(class_id=cls, class_name=name,
                       prototype_initialized=int(gate.target_valid_proto[cls]))
            for output, source in (('predicted_count', 'candidate_count'),
                                   ('confidence_pass_count', 'confidence_pass_count'),
                                   ('weight_pass_count', 'weight_pass_count'),
                                   ('joint_selected_count', 'selected_count'),
                                   ('updated_count', 'updated_count')):
                row[output] = int(diag[prefix + source])
            proto.append(row)
    phy_groups, phy_losses, downweighted, phy_count = {}, {}, 0, 0
    if flags['phy'] and (not branch_mode or model.mechanism_cfg.get('detailed_output')):
        vegetation = model._class_mask(label, model.phy_positive_classes)
        response = r_phy[:, 0] if r_phy.dim() == 4 else r_phy
        for name, mask in (('vegetation', vegetation), ('nonvegetation', ~vegetation),
                           ('goji', label.eq(0)), ('othervegetation', label.eq(1))):
            mask = mask & valid
            phy_groups['target_' + name + '_pred'] = (
                response.detach()[mask].float().sum().item(), int(mask.sum()))
        for name, pair in getattr(model, 'mechanism_phy_source', {}).items():
            phy_groups['source_' + name] = pair
        count_mask = valid & vegetation
        if hphy is not None and w_base is not None:
            count_mask = count_mask & w_base.gt(0)
        phy_count = int(count_mask.sum())
        if w_base is not None:
            downweighted = int((valid & vegetation & w_t_phy.detach().lt(w_base)).sum())
        if log_vars is not None and 'loss_phy_src' in log_vars:
            phy_losses['loss_phy_src'] = log_vars['loss_phy_src']
    branch = {}
    def statistic(name, value, mask):
        if value.dim() == 4:
            value = value[:, 0]
        branch[name] = (value.detach()[mask].float().sum().item(), int(mask.sum()))
    if branch_mode:
        if flags['phy']:
            mask = valid & model._class_mask(label, model.phy_apply_to_classes)
            statistic('phy_response_mean_valid', r_phy, mask)
            statistic('phy_weight_before_mean', w_base, mask)
            statistic('phy_weight_after_mean', w_t_phy, mask)
            ratio_mask = mask if hphy is None else mask & w_base.gt(0)
            branch['phy_downweight_ratio'] = (int((ratio_mask & w_t_phy.detach().lt(w_base)).sum()), int(ratio_mask.sum()))
            if hphy is not None:
                supported = mask & w_base.gt(0)
                for name in ('r_veg', 'q_goji_given_veg', 's_phy_class_agreement', 'g_phy'):
                    statistic('hphy_' + name + '_mean', hphy[name], supported)
                statistic('hphy_attenuation_mean', 1 - hphy['g_phy'], supported)
        if flags['struct']:
            mask = valid & model._class_mask(label, model.struct_gate.struct_positive_classes)
            statistic('struct_response_mean_valid', structural['R_struct'], mask)
            statistic('struct_weight_before_mean', w_t_phy if flags['phy'] else w_base, mask)
            statistic('struct_weight_after_mean', w_t_ps, mask)
        if flags['proto']:
            statistic('proto_weight_before_mean', w_t_ps, valid)
            statistic('proto_weight_after_mean', final, valid)
        if flags['edge']:
            mask = valid & model.mechanism_boundary(label)[:, 0]
            statistic('edge_response_mean_valid', edge_response, mask)
            statistic('edge_supervision_mean_valid', boundary_weight, mask)
    return dict(iter=model.local_iter + 1, valid_count=count, total_count=valid.numel(),
        high_conf_count=int((valid & confidence.ge(model.pseudo_threshold)).sum()),
        sums=sums, classes=proto, modules=flags, states=states, branch=branch,
        phy_groups=phy_groups, phy_losses=phy_losses,
        phy_parameters={key: getattr(model, key) for key in
            ('phy_target_supervision_mode', 'tau_conf_phy', 'phy_pos_weight', 'phy_neg_weight',
             'lambda_phy_src', 'lambda_phy_tgt')} if flags['phy'] else {},
        phy_downweighted_count=downweighted, phy_vegetation_count=phy_count,
        tau_proto=gate.tau_proto if flags['proto'] else '',
        tau_conf_proto=gate.tau_conf_proto if flags['proto'] else '',
        proto_valid_class_count=int(gate.valid_proto.sum()) if flags['proto'] else '',
        proto_update_active=int(gate.target_update and model.local_iter >= gate.target_update_start)
            if flags['proto'] else '',
        proto_weight_active=int(states['proto'] == 'active') if flags['proto'] else '')


def append_unique_csv(path, rows, keys):
    """Atomic small-file append with duplicate and changed-state detection."""
    path = Path(path)
    existing = []
    if path.exists():
        with path.open(newline='', encoding='utf-8') as stream:
            existing = list(csv.DictReader(stream))
    index = {tuple(str(row[key]) for key in keys): row for row in existing}
    for row in rows:
        serialized = {key: str(value) for key, value in row.items()}
        key = tuple(serialized[name] for name in keys)
        if key in index:
            if index[key] != serialized:
                raise ValueError('Conflicting existing CSV node: ' + str(key))
            continue
        existing.append(serialized)
        index[key] = serialized
    if not existing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(existing[0]))
        writer.writeheader()
        writer.writerows(existing)
    temporary.replace(path)


FIXED_TARGET_PROBES = {
    'goji30IRRG2goji26IRRG': ('26', 'img_dir', '3_768_964_1024_1220.tif'),
    'goji26IRRG2goji30RGB': ('30', 'rgb_dir', '3_1536_640_1792_896.tif'),
    'goji26RGB2goji30RGB': ('30', 'rgb_dir', '3_1024_1280_1280_1536.tif')}


def exact_probe_index(dataset, filename):
    root = Path(dataset.img_dir).resolve()
    matches = []
    for index, info in enumerate(dataset.img_infos):
        if Path(info['filename']).name != filename:
            continue
        path = (root / info['filename']).resolve()
        if root not in path.parents:
            raise ValueError('Fixed probe escapes the target image directory: ' + str(path))
        matches.append((index, path))
    if len(matches) != 1:
        raise ValueError('Expected exactly one target probe {} in {}; found {}'.format(
            filename, root, len(matches)))
    index, path = matches[0]
    if not path.is_file():
        raise FileNotFoundError('Required target probe does not exist: ' + str(path))
    return index


def deterministic_probe_config(cfg, allow_daformer=False):
    target = deepcopy(dict(cfg['data']['train']['target']))
    if allow_daformer and target.get('use_indices',False):
        # Replay's shared sample interface needs the separate VI tensor as well
        # as the historical six-channel image. This flag only adds that return.
        target['use_vi'] = True
    norm = deepcopy(dict(cfg['target_img_norm_cfg']))
    if allow_daformer:
        # Inherited top-level fields can retain a six-channel order after the
        # actual training pipeline has been replaced with RGB-only transforms.
        normalizers = [step for step in target.get('pipeline',[]) if step.get('type')=='Normalize']
        if len(normalizers)>1:
            raise ValueError('Baseline replay requires one historical target Normalize step')
        if normalizers:
            norm = deepcopy(dict(normalizers[0]))
            norm.pop('type')
    size = tuple(cfg.get('crop_size', (256, 256)))
    margins = target.pop('crop_pseudo_margins', [0, 0, 0, 0])
    target.update(test_mode=True, ann_dir=None, pipeline=[
        dict(type='LoadImageFromFile'), dict(type='LoadGojiIndices'),
        dict(type='Resize', img_scale=size, keep_ratio=False),
        dict(type='Normalize', **norm), dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img','vi_img'] if target.get('use_vi',False) else ['img'], meta_keys=[
            'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'img_norm_cfg'])])
    result = dict(dataset=target, index=int(cfg['uda'].get('fixed_probe_index', 0)),
                margins=margins, size=size,
                channel_order=cfg['uda'].get('target_input_channel_order',cfg['uda'].get('input_channel_order',[])))
    architecture = r'(?:hrda_?3vi|daformer_3vi|daformer_r101_no3vi)' if allow_daformer else r'hrda_?3vi'
    match = re.fullmatch(r'(?:doris_)?(goji\w+2goji\w+)_' + architecture +
                        r'(?:_phy_gate|_struct_gate|_phy_struct_gate|_ps_proto|_struct_edge|'
                        r'_pbcr|_proto_struct_edge)?(?:_hphy15k|_hphy_early3k|_hphy_early4k)?', cfg.get('name', ''))
    if match and match[1] in FIXED_TARGET_PROBES:
        direction = match[1]
        domain, view, filename = FIXED_TARGET_PROBES[direction]
        validation = cfg['data']['val']
        target_root = Path(target['data_root']).resolve()
        validation_root = Path(validation['data_root']).resolve()
        if (target_root.name != domain or validation_root != target_root or
                Path(target['img_dir']).parts[-2] != view or
                Path(validation['img_dir']).parts[-2] != view):
            raise ValueError('Fixed probe target domain/view differs from ' + direction)
        # These requested scenes are in val. No annotations or evaluation are loaded.
        target.update(data_root=validation['data_root'], img_dir=validation['img_dir'],
                      split=validation.get('split'))
        target['pipeline'] = [step for step in target['pipeline'] if step['type'] != 'Resize']
        result.update(direction=direction, filename=filename, index=None, size=None,
                      expected_path=str((validation_root / target['img_dir'] / filename).resolve()),
                      selection_policy='exact_target_val_native_v1')
    return result


def load_probe_sample(probe_cfg, model):
    """Load the shared deterministic sample without creating training artifacts."""
    from mmseg.datasets import build_dataset
    with readonly_probe(model):
        dataset = build_dataset(probe_cfg['dataset'])
        if probe_cfg.get('filename'):
            index = exact_probe_index(dataset, probe_cfg['filename'])
        else:
            indices = sorted(range(len(dataset.img_infos)), key=lambda i: dataset.img_infos[i]['filename'])
            index = indices[probe_cfg['index']]
        data = dataset[index]
        image, metas = data['img'].data, data['img_metas'].data
        vi = data['vi_img'].data if 'vi_img' in data else image.new_empty((0,*image.shape[-2:]))
        metas.update(flip=False, flip_direction=None)
        if probe_cfg.get('filename'):
            if Path(metas['filename']).resolve() != (Path(dataset.img_dir) /
                    dataset.img_infos[index]['filename']).resolve():
                raise ValueError('Loaded probe path differs from the exact target match')
            if tuple(metas['ori_shape'][:2]) != tuple(image.shape[-2:]):
                raise ValueError('Native probe input differs from the original TIF pixel grid')
            channels = len(metas['img_norm_cfg']['mean'])
            if image.shape[0] not in (3,6) or image.shape[0]!=channels:
                raise ValueError('Fixed probe channels differ from the historical image normalization')
            if vi.shape[0] not in (0,3) or vi.shape[-2:]!=image.shape[-2:]:
                raise ValueError('Fixed probe VI channels differ from the native image grid')
            if model.enable_phy and vi.shape[0]!=3:
                raise ValueError('Enabled physiology requires three real VI channels')
            if not torch.isfinite(image).all() or not torch.isfinite(vi).all():
                raise ValueError('Non-finite fixed probe input')
    sample = (image.unsqueeze(0), vi.unsqueeze(0), [metas])
    digest = hashlib.sha256(image.numpy().tobytes() + vi.numpy().tobytes()).hexdigest()
    description = dict(probe_cfg, sample_id=Path(metas['filename']).name,
        image_shape=list(image.shape), tensor_sha256=digest,
        vi_shape=list(vi.shape),vi_available=bool(vi.numel()),
        spatial_shape=list(image.shape[-2:]), crop=None,
        data_source='fixed_probe_recomputed', preprocessing='deterministic resize + target normalization')
    if probe_cfg.get('filename'):
        description.update(target_path=str(Path(metas['filename']).resolve()),
            original_shape=list(metas['ori_shape'][:2]),
            preprocessing='native pixel grid + training target channel order / VI / normalization')
    else:
        description['filename'] = metas['filename']
    return sample, description


def hphy_gate_metadata(model, score, gate, weight, valid, label):
    """Summarize the actual positive-weight physiology scope on the full grid."""
    from mmseg.models.uda.phy_reliability import phy_gate_strength
    supported = valid.astype(bool) & (weight > 0)
    active = supported & np.isin(label, [0, 1]) & (gate < 1)
    attenuation = 1 - gate
    params = model.phy_gate_params
    rho = phy_gate_strength(model.local_iter, **params) if model.use_target_gate else 0.
    reason = (None if active.any() else
              'target gate disabled' if not model.use_target_gate else
              'no valid positive base weights' if not supported.any() else
              'no Teacher Goji/Vegetation pixels with positive base weight' if not
              (supported & np.isin(label, [0, 1])).any() else
              'zero schedule strength or neutral physiology agreement')
    return dict(gate_schedule=params['schedule'],
        gate_initial_strength=params.get('initial_strength'),
        gate_full_iter=params.get('full_iter'), gate_rho_at_checkpoint=rho,
        gate_min_weight=model.phy_min_weight,
        gate_active_pixel_ratio=float(active.sum() / max(1, supported.sum())),
        gate_active_pixel_count=int(active.sum()),
        gate_ratio_denominator='valid pixels with W_before > 0',
        mean_S_phy_on_active_pixels=float(score[active].mean()) if active.any() else None,
        mean_A_phy_on_active_pixels=float(attenuation[active].mean()) if active.any() else None,
        max_A_phy_on_active_pixels=float(attenuation[active].max()) if active.any() else None,
        gate_empty_reason=reason)


def branch_probe_payload(model, image, vi, metas, valid, metadata, unavailable_weights=()):
    """Capture the same enabled branch tensors for online collection and replay."""
    maps = probe_arrays(model, image, vi, metas, valid)
    for key in unavailable_weights:
        maps.pop(key, None)
    metadata['tensor_shapes'] = {key:list(value.shape) for key,value in maps.items()
                                 if value.ndim == 2}
    metadata['teacher_logits_shape'] = maps['teacher_logits_shape'].tolist()
    if 'prototype_feature_shape' in maps:
        metadata['prototype_feature_shape'] = maps['prototype_feature_shape'].tolist()
    from tools.doris_probe_data import spatial_statistics
    metadata['statistics'] = spatial_statistics(maps)
    diagnostics = dict(valid_pixel_ratio=float(maps['valid_mask'].mean()), all_finite=True)
    mask = maps['valid_mask'].astype(bool)
    if 'delta_w_proto' in maps:
        delta = maps['delta_w_proto'][mask]
        diagnostics['prototype_difference'] = dict(
            max_abs=float(np.abs(delta).max()) if delta.size else 0,
            mean_abs=float(np.abs(delta).mean()) if delta.size else 0,
            changed_valid_ratio=float(np.count_nonzero(delta)/delta.size) if delta.size else 0)
    metadata['diagnostics'] = diagnostics
    aliases = {'w_before_phy':'w_base', 'w_after_phy':'w_t_phy',
               'w_before_proto':'w_pre_proto', 'w_after_proto':'w_t_proto'}
    kept = ('valid_mask','pseudo_label','w_conf','rule_iter','proto_active',
            'r_phy','m_phy','delta_w_phy','r_conf','r_loc','r_adj','r_prob','r_struct',
            'adjacency_available','semantic_available','w_before_struct','w_after_struct',
            'delta_w_proto','teacher_pseudo_boundary','r_edge','edge_supervision')
    payload = {key:maps[key] for key in kept if key in maps}
    if 'r_phy' in maps:
        metadata['head_version'] = getattr(model, 'phy_mode', 'legacy_binary')
    if 'r_veg' in maps:
        for key in ('r_veg', 'q_goji_given_veg', 'p_goji', 'p_other_veg', 'p_nonveg',
                    's_phy_class_agreement', 'g_phy', 'a_phy', 'attenuation_phy',
                    'teacher_label', 'teacher_confidence', 'teacher_top2_margin',
                    'vi_ndvi', 'vi_ndre', 'vi_cire'):
            payload[key] = maps[key]
        metadata['r_phy_alias'] = 'r_veg; class-conditioned gating uses s_phy_class_agreement'
        metadata['hphy_cfg'] = dict(model.phy_cfg)
        metadata.update(hphy_gate_metadata(model, maps['s_phy_class_agreement'],
            maps['g_phy'], maps['w_before_phy'], maps['valid_mask'], maps['pseudo_label']))
        code_root = Path(__file__).resolve().parents[2]
        metadata.setdefault('source_sha256', {}).update({
            name: hashlib.sha256((code_root / name).read_bytes()).hexdigest()
            for name in ('mmseg/models/uda/phy_reliability.py', 'mmseg/models/uda/dacs.py')})
        metadata.update(
            label_mapping={str(i): name for i, name in enumerate(model.class_names)},
            ignore_index=255, vi_order=['NDVI', 'NDRE', 'CIRE'],
            work_dir=getattr(model, 'train_cfg', {}).get('work_dir'),
            prediction_identity='EMA Teacher; Student validation is a separate output',
            relative_attenuation_statistics='valid pixels with w_before_phy > 0')
    if 'r_phy' in maps:
        metadata['phy_weight_definitions'] = dict(
            w_conf='EMA Teacher maximum softmax probability at each pixel',
            w_before_phy='DACS image-level high-confidence fraction times valid mask',
            w_after_phy='Before weight multiplied by the scheduled physiology gate',
            delta_w_phy='After weight minus before weight')
    for destination, source in aliases.items():
        if (destination.endswith('_phy') and 'r_phy' in maps) or (
                destination.endswith('_proto') and 'w_t_proto' in maps):
            payload[destination] = maps[source]
    norm = metas[0]['img_norm_cfg']
    view = image[0,:3].detach().float().cpu().numpy().transpose(1,2,0)
    payload['target_image'] = np.clip((view*np.asarray(norm['std'][:3]) +
        np.asarray(norm['mean'][:3]))/255,0,1).astype(np.float32)
    if metadata['modules'] == dict(phy=False, struct=True, proto=False, edge=False):
        from tools.doris_struct_evidence import prepare_struct_evidence, struct_statistics
        metadata['struct_cfg'] = dict(model.struct_cfg, eps=model.struct_gate.eps)
        metadata['struct_gate_strength'] = model.struct_gate.gate_strength(model.local_iter)
        metadata['struct_probability_dtype'] = 'float16' if image.is_cuda else 'float32'
        with readonly_probe(model):
            prediction = model.get_model()(img=[image], img_metas=[metas], return_loss=False, rescale=True)
        if len(prediction) != 1 or np.asarray(prediction[0]).shape != (256, 256):
            raise ValueError('Structural Student test prediction must cover the native 256 grid')
        predicted_classes = np.asarray(prediction[0])
        if predicted_classes.dtype.kind not in 'iu' or not np.isin(predicted_classes, [0, 1, 2, 3, 4]).all():
            raise ValueError('Structural Student test output must contain class IDs 0..4')
        payload['struct_prediction'] = predicted_classes.astype(np.uint8, copy=True)
        metadata['prediction_identity'] = 'Student test-mode prediction'
        metadata['student_inference'] = student_test_metadata(model, image, metas)
        metadata['prediction_checkpoint'] = dict(path=metadata.get('checkpoint'),
            sha256=metadata.get('checkpoint_sha256'), state_sha256=metadata.get('state_sha256'),
            iteration=metadata.get('iter'), capture_mode=metadata.get('capture_mode'))
        if 'w_before_struct' in payload and 'w_after_struct' in payload:
            for key in ('confidence_available', 'continuity_available', 'evidence_count',
                        'target_cross_neighbor_count', 'source_cross_prior_available'):
                if key in maps:
                    payload[key] = maps[key]
            enriched, metadata = prepare_struct_evidence(payload, metadata)
            payload.update(enriched)
            metadata['tensor_shapes'].update({key: list(value.shape) for key, value in enriched.items() if value.ndim == 2})
            if metadata['struct_validation']['passed']:
                metadata['struct_statistics'] = struct_statistics(enriched, metadata)
    payload['metadata'] = np.array(json.dumps(metadata,sort_keys=True))
    return payload


class MechanismCollector:
    """Invoked by the existing PrecisionLoggerHook, not a second training Hook."""

    def __init__(self, runner, model, probe_filename=None, analysis_root=None):
        self.model, self.runner = model, runner
        self.modules = module_flags(model)
        if not any(self.modules.values()):
            raise ValueError('Baseline must disable module mechanism collection')
        self.cfg = model.mechanism_cfg
        self.output_mode = self.cfg.get('output_mode', 'branches')
        if self.output_mode not in ('branches', 'detailed'):
            raise ValueError('mechanism_output_mode must be branches or detailed')
        self.physiology = self.cfg.get('physiology', False) and self.modules['phy']
        if self.cfg['make_figures'] and not self.cfg['save_raw']:
            raise ValueError('Automatic summary figures require mechanism_save_raw=True')
        if self.cfg.get('online_plot', False) and not (self.cfg['probe_enabled'] and self.cfg['save_raw']):
            raise ValueError('Online PNGs require fixed_probe_enabled and mechanism_save_raw')
        self.base_root = (Path(analysis_root) if analysis_root is not None else
                          Path(runner.work_dir) / 'analysis').resolve()
        self.root = self.base_root
        pointer = self.base_root / 'active_resume.json'
        if pointer.exists():
            selected = (self.base_root / json.loads(pointer.read_text(encoding='utf-8'))['path']).resolve()
            if (selected != self.base_root and self.base_root not in selected.parents) or not selected.is_dir():
                raise ValueError('Invalid active resume analysis directory')
            self.root = selected
        previous = self.root / 'manifest.json'
        if previous.exists() and self.output_mode == 'branches' and json.loads(
                previous.read_text(encoding='utf-8')).get('output_mode') != 'branches':
            self.root = self.root / 'branches_v4'
        elif previous.exists() and json.loads(previous.read_text(encoding='utf-8')).get('schema_version', 1) < 2:
            # Preserve already-collected Full-only data when upgrading schemas.
            self.root = self.root / 'module_collection_v2'
        self.pending = []
        self.window_start = runner.iter
        self.sample = None
        self.extra_probes = []
        effective = getattr(runner, 'mechanism_config', None)
        if effective is None:
            from mmcv import Config
            effective = Config.fromstring(runner.meta['config'], '.py')
        config = (effective._cfg_dict.to_dict() if hasattr(effective, '_cfg_dict')
                  else deepcopy(dict(effective)))
        self.source_cfg = deepcopy(config.get('data', {}).get('train', {}).get('source', {}))
        identity = deepcopy(config)
        for key in ('resume_from', 'load_from', 'gpu_ids', 'work_dir'):
            identity.pop(key, None)
        identity['model'].get('train_cfg', {}).pop('work_dir', None)
        identity.get('uda', {}).pop('model', None)  # runtime duplicate of model
        # Output-only additions must not invalidate existing schema-4 runs on resume.
        for key in ('mechanism_online_plot', 'mechanism_strict'):
            identity.get('uda', {}).pop(key, None)
        if identity.get('uda', {}).pop('fixed_probe_filenames', None) is not None:
            identity['uda']['fixed_probe_max_samples'] = 1
        identity.get('checkpoint_config', {}).pop('meta', None)
        self.identity = json_digest(identity)
        self.probe_cfg = deterministic_probe_config(config)
        if probe_filename is not None:
            self.probe_cfg.update(filename=probe_filename, expected_path=str((
                Path(self.probe_cfg['dataset']['data_root']) /
                self.probe_cfg['dataset']['img_dir'] / probe_filename).resolve()))
        if self.probe_cfg.get('filename'):
            current_sample = self.root / 'fixed_probe/sample_manifest.json'
            old = json.loads(current_sample.read_text(encoding='utf-8')) if current_sample.exists() else {}
            if probe_filename is None and any(old.get(key) != self.probe_cfg[key] for key in
                   ('direction', 'filename', 'selection_policy')):
                sample_tag = self.probe_cfg['direction'] + '__' + Path(self.probe_cfg['filename']).stem + '_native_v1'
                self.root = self.base_root / 'fixed_samples' / sample_tag
            if config['model'].get('feature_scale') == 'all':
                raise ValueError('Native fixed probes require full-image prototype features, not cached training crops')
        filenames = config['uda'].get('fixed_probe_filenames', [])
        if probe_filename is None and filenames:
            if (filenames[0] != self.probe_cfg.get('filename') or
                    len(set(filenames)) != len(filenames) or
                    len(filenames) > config['uda'].get('fixed_probe_max_samples', 1) or
                    any(Path(name).name != name or Path(name).suffix.lower() != '.tif' for name in filenames)):
                raise ValueError('Fixed probe filenames must be unique TIFF basenames, start with the '
                                 'original fixed sample, and fit fixed_probe_max_samples')
        elif probe_filename is None and config['uda'].get('fixed_probe_max_samples', 1) != 1:
            raise ValueError('Multiple fixed probes require explicit fixed_probe_filenames')
        self.manifest = dict(schema_version=4 if self.output_mode == 'branches' else 3 if self.physiology else 2,
            output_mode=self.output_mode, detailed_output=self.cfg.get('detailed_output', False),
            experiment_name=config.get('name', Path(runner.work_dir).name),
            physiology_compact=self.physiology, modules=self.modules, config_sha256=self.identity,
            git_commit=runner.meta.get('git_commit'), seed=config.get('seed'),
            backbone=config['model'].get('backbone', {}).get('type'),
            input_channels=config['model'].get('backbone', {}).get('in_chans',
                config['model'].get('backbone', {}).get('in_channels')),
            max_iters=config.get('runner', {}).get('max_iters'),
            phy_warmup_start=model.phy_warmup_iters if self.modules['phy'] else None,
            phy_rampup_iters=model.phy_rampup_iters if self.modules['phy'] else None,
            struct_warmup_start=model.struct_gate.struct_warmup_iters if self.modules['struct'] else None,
            data_source='training_batch_actual', sample_interval=self.cfg['sample_interval'],
            summary_interval=self.cfg['summary_interval'], probe_interval=self.cfg['probe_interval'],
            proto_weight_start=model.proto_gate.weight_start if self.modules['proto'] else None,
            proto_target_update_start=model.proto_gate.target_update_start if self.modules['proto'] else None)
        if self.modules['phy'] and getattr(model, 'phy_mode', 'legacy_binary') == 'hierarchical_gv':
            self.manifest['phy_gate'] = dict(model.phy_gate_params)
        if self.modules['struct'] and model.struct_gate.fusion_mode == 'struct_distinct_v1':
            self.manifest['struct_cfg'] = dict(model.struct_cfg)
            self.manifest['struct_gate_initial_strength'] = model.struct_gate.gate_strength(0)
            self.manifest['struct_warmup_start'] = (
                0 if model.struct_gate.struct_gate_schedule == 'early_nonzero_linear'
                else model.struct_gate.struct_warmup_iters)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / 'manifest.json'
        if path.exists():
            previous_manifest = json.loads(path.read_text(encoding='utf-8'))
            stable = {k: v for k, v in previous_manifest.items()
                      if k not in ('snapshots', 'snapshot_policy', 'resume_origin')}
            if stable != self.manifest:
                raise ValueError('Existing analysis manifest differs; preserve it and use a separate work_dir')
            self.manifest.update({k: previous_manifest[k] for k in ('snapshots', 'snapshot_policy', 'resume_origin')
                                  if k in previous_manifest})
        if self.cfg.get('online_plot', False):
            self.manifest['snapshot_policy'] = dict(online_plot=True, strict=self.cfg.get('strict', True),
                                                    interval=self.cfg['probe_interval'])
        if getattr(runner, 'resume_from', None):
            self.prepare_resume(runner.iter)
        self.save_manifest()
        self.write_definitions()
        if self.cfg['probe_enabled']:
            self.load_sample()
            if probe_filename is None:
                for filename in filenames[1:]:
                    sample_tag = self.probe_cfg['direction'] + '__' + Path(filename).stem + '_native_v1'
                    self.extra_probes.append(MechanismCollector(runner, model, filename,
                        self.base_root / 'fixed_samples' / sample_tag))
        if self.probe_cfg.get('filename'):
            pointer = self.base_root / 'active_resume.json'
            temporary = pointer.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(dict(path=self.root.relative_to(self.base_root).as_posix())),
                                 encoding='utf-8')
            temporary.replace(pointer)
        runner.logger.info('Mechanism collection: sample every %s, CSV/probe every %s/%s; '
            'tau_proto=%s, tau_conf_proto=%s; outputs=%s', self.cfg['sample_interval'],
            self.cfg['summary_interval'], self.cfg['probe_interval'], model.proto_gate.tau_proto if self.modules['proto'] else 'disabled',
            model.proto_gate.tau_conf_proto if self.modules['proto'] else 'disabled', str(self.root))

    def prepare_resume(self, completed):
        """Keep a rollback's new observations separate from its abandoned future."""
        tables = []
        has_future = any(int(step) > completed for step in self.manifest.get('snapshots', {}))
        for path in sorted((self.root / 'metrics').glob('*.csv')):
            with path.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            if fields and 'iter' in fields:
                has_future |= any(int(row['iter']) > completed for row in rows)
            tables.append((path, fields, rows))
        arrays = []
        for folder in (self.root / 'fixed_probe', self.root / 'physiology/arrays'):
            for path in sorted(folder.rglob('iter_*')):
                match = re.match(r'iter_(\d+)(?:_|\.)', path.name)
                if path.is_file() and path.suffix in ('.npz', '.png', '.pdf') and match:
                    step = int(match.group(1))
                    arrays.append((path, step))
                    has_future |= step > completed
        if not has_future:
            return

        source = self.root
        staging = Path(tempfile.mkdtemp(prefix='.resume_', dir=self.base_root))
        target = self.base_root / 'resume_runs' / ('from_{:06d}_{}'.format(completed, uuid.uuid4().hex[:8]))
        # Verify both computed directories before moving the staged directory on Windows.
        for directory in (staging, target):
            if self.base_root.resolve() not in directory.resolve().parents:
                raise ValueError('Resume analysis must stay inside the work_dir analysis directory')
        for path, fields, rows in tables:
            destination = staging / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if fields and 'iter' in fields:
                with destination.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(row for row in rows if int(row['iter']) <= completed)
            else:
                shutil.copy2(path, destination)
        static_files = [source / 'fixed_probe/sample_manifest.json']
        for folder in (source / 'reports', source / 'physiology/data'):
            static_files.extend(path for path in folder.rglob('*') if path.is_file())
        for path in static_files + [path for path, step in arrays if step <= completed]:
            if path.is_file():
                destination = staging / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        self.manifest['snapshots'] = {step: entry for step, entry in self.manifest.get('snapshots', {}).items()
                                      if int(step) <= completed}
        self.manifest['resume_origin'] = dict(source_analysis=source.relative_to(self.base_root).as_posix(),
            checkpoint=str(self.runner.resume_from), completed_iter=completed)
        self.root = staging
        self.save_manifest()
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
        self.root = target
        pointer = self.base_root / 'active_resume.json'
        temporary = pointer.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(dict(path=target.relative_to(self.base_root).as_posix()), indent=2), encoding='utf-8')
        temporary.replace(pointer)
        self.runner.logger.info('Resume at iter=%d has later historical observations; '
            'continuing analysis at %s, inherited evidence through iter=%d from %s',
            completed, str(target), completed, str(source))

    def save_manifest(self):
        path = self.root / 'manifest.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.manifest, indent=2), encoding='utf-8')
        temporary.replace(path)

    def record_snapshot(self, completed, **values):
        snapshots = self.manifest.setdefault('snapshots', {})
        entry = dict(snapshots.get(str(completed), {}))
        if entry.get('snapshot_status') == 'completed' and values.get('snapshot_status') == 'failed':
            entry['last_failed_attempt'] = dict(values)
            snapshots[str(completed)] = entry
            self.save_manifest()
            return
        if values.get('snapshot_status') == 'in_progress':
            entry = {}
        entry.update(iter=completed, data_source='fixed_probe_recomputed', **values)
        snapshots[str(completed)] = entry
        self.save_manifest()

    def write_definitions(self):
        reports = self.root / 'reports'
        reports.mkdir(exist_ok=True)
        metric = '''# Mechanism data definitions
data_source=training_batch_actual: sampled tensors from real training batches.
sample_interval and n_sampled_batches declare sampling; this is NOT an all-500-step mean.
iter is completed runner iterations. window_start may be a partial resume interval.
valid_count sums valid image pixels after ignore/crop masks; valid_area_ratio uses all image pixels.
teacher_high_conf_ratio counts confidence>=pseudo_threshold within valid pixels.
All *_mean_valid fields sum numerators and divide by accumulated valid_count.
w_conf is Teacher maximum probability, NOT the DACS batch-wide high-confidence ratio.
w_t_ps is after MPARC and before prototype weighting; w_t_proto is the final training weight.
boundary_weight_mean_valid is the unscaled actual target boundary weight including domain coefficient/ramp.
proto_target_update_ratio=updated feature pixels / all valid predicted feature pixels.
proto_valid_class_count is the final sampled fused-cache class count, not an interval average.
proto_update_active/proto_weight_active distinguish inactivity from zero numerical values.
Prototype CSV counts use the actual feature grid; spatial support follows bilinear w_t_ps alignment.
predicted_count excludes ignore and wholly spatial-invalid feature cells.
confidence_pass_count, weight_pass_count and joint_selected_count precede min_pixels gating.
updated_count additionally requires min_pixels and the target-update schedule.
selected_ratio=joint_selected_count/predicted_count for the same class.
selected_ratio_all_valid uses the sum of valid feature predictions over all classes.
prototype_initialized is the final sampled target-cache state; no GT is read.
Zero denominators yield 0 plus explicit zero counts; they are not evidence of measured accuracy.
Fixed arrays are fixed_probe_recomputed, never the random training batch at that iteration.
memory is process-lifetime peak CUDA allocated MiB (including probes), not reserved or nvidia-smi memory.
probe_cost run_peak_increase_mib is new lifetime peak growth, not an isolated probe peak.
train_iter_time is training-step wall time excluding data loading/validation/fixed-probe rendering.
No forced CUDA synchronization is added solely to training timing.
'''
        mapping = '''# Formula code mapping
R_conf / W_conf -> Teacher max probability -> r_conf / w_conf.
R_loc -> existing R_continuity; R_prob -> existing R_sem.
R_adj and R_prob use source probability/adjacency priors, not PBCR feature prototypes.
R_struct -> existing struct_gate fusion; no formula or coefficient changes in this collection task.
W_t_PS -> DACS w_t_ps snapshot before ProtoGate.compute_weight.
W_t_proto -> DACS final_gate_pseudo_weight before ClassMix; never collect the in-place mixed weights.
R_edge -> existing doris_edge_responses target fusion.
Disabled module fields are absent in NPZ and blank in CSV, not zero-valued evidence.
W_pre_proto is the actual input of ProtoGate: W_base for Prototype-only, not MPARC.
W_base is DACS batch-wide confidence-ratio weight after spatial masks.
Modules/states distinguish disabled, warmup, cache_unready, ramping and active.
Enabled modules determine which plots/CSVs are saved; no disabled module is evaluated.
tau_proto / tau_conf_proto -> actual ProtoGate.update_target conditions, unchanged.
Loss keys remain original MMSeg keys; HRDA fused/hr branches do not represent additional innovations.
source loss_phy_src is raw; total loss includes lambda_phy_src times this raw value.
Fixed probe is eval/no_grad; structure update_priors=False and prototype read_only=True.
Fixed rule_iter is the current model counter after completed iter; not a replay of that step's batch.
R_phy is vegetation-related physiological reliability, NOT Goji probability.
M_phy is the Teacher vegetation-class mask; W_before_phy is the spatial-masked DACS weight.
Actual gate: a=.82+.18*R_phy; a is halved when R_phy<.30 (soft filter).
Warmup/ramp: rho=clip((rule_iter-6000)/4000,0,1); W_phy=W_before*((1-M_phy)+M_phy*((1-rho)+rho*a)).
This multiplicative gate is not old PDF Eq.(8)'s class-conditional replacement formula.
legacy target physiology BCE uses only Teacher vegetation predictions above pseudo_threshold.
balanced_high_conf uses valid, non-ignore Teacher labels: pos={0,1}, neg={2,3,4}, confidence>=tau_conf_phy.
L_phy_tgt=.008*(.5*mean_BCE_pos+.5*mean_BCE_neg); renormalize weights over nonempty groups.
If both groups are empty, return a differentiable zero; no extra head forward or backward.
CSV source response groups use source GT; target groups use Teacher labels, never target GT.
phy_downweight_ratio=actually_downweighted_vegetation_count/valid_Teacher_vegetation_count.
phy loss columns are sampled-batch scalar means (source raw, target lambda-weighted), not new train losses.
'''
        if getattr(self.model, 'phy_mode', None) == 'hierarchical_gv':
            mapping = '''# H-Phy V1 formula code mapping
VI encoder -> r_veg and q_goji_given_veg, each independently sigmoid activated.
p_goji=r_veg*q_goji_given_veg; p_other_veg=r_veg*(1-q_goji_given_veg); p_nonveg=1-r_veg.
s_phy_class_agreement selects p_goji or p_other_veg by Teacher argmax.
g_phy=1-rho*(1-phy_min_weight)*(1-s_phy_class_agreement) on valid vegetation predictions.
w_after_phy=w_before_phy*g_phy; attenuation_phy=1-g_phy, restricted to positive base weights.
Source raw=(BCE_veg+beta_gv*balanced_BCE_gv)/(1+beta_gv); weighted=lambda_phy_src*raw.
Target weighted=lambda_phy_tgt*balanced_BCEWithLogits_veg; direct target GV loss is zero.
Source loss_phy_src_*_raw and loss_phy_src_weighted have explicit units. Legacy loss_phy_src is raw.
Target GT is available only to the independent offline reference analyzer.
r_phy is a compatibility alias of r_veg. H-Phy figures use s_phy_class_agreement and attenuation_phy.
'''
            mapping += '\nGate strength uses phy_gate_strength(local_iter, **phy_gate_params).\n'
            mapping += 'Effective saved schedule parameters = ' + json.dumps(self.model.phy_gate_params, sort_keys=True) + '\n'
        for name, content in (('metric_definitions.md', metric), ('formula_code_mapping.md', mapping)):
            path = reports / name
            if (self.output_mode == 'detailed' or self.cfg.get('detailed_output', False)) and not path.exists():
                path.write_text(content, encoding='utf-8')
        branch_mapping = '''# Branch evidence and ablation mapping
Baseline -> Physiology / Structural / Prototype / Boundary: individual branch contribution.
Baseline -> MPARC / PBCR: module-level joint effect. MPARC/PBCR -> Full: complementarity.
Only compare within a direction and matching protocols. Heatmaps do not establish IoU gains.
branch_mechanism_500.csv is training_batch_actual, sampled every 100, summarized every 500.
Each metric sums numerators and denominators over active sampled batches, not all 500 iterations.
phy scope: valid Teacher vegetation predictions. struct scope: valid Teacher structural classes 2/3/4.
proto weights: all valid Teacher image pixels; selection/update counts: valid feature-grid candidates.
edge scope: valid Teacher semantic boundary pixels, using the exact training label_to_edge rule.
NA means disabled, not yet active, no valid denominator, or no available sampled batch; never a fake zero.
The state and *_sampled_batches columns distinguish warmup/cache_unready from observed zero change.
Prototype selection/update metrics follow target_update_start; weights follow actual cache/weight activation.
Raw fixed probes are fixed_probe_recomputed and include true before/after weights during warmup.
capture_mode=online_after_train_iter identifies a new forward at that node's live model/cache state.
The legacy data_source name does not mean a final checkpoint was replayed at past iterations.
Offline rendering reads saved arrays only; it never computes missing historical nodes.
Native fixed probes use exact target val filenames and preserve the full original TIF pixel grid.
Spatial PNGs under snapshots/spatial/iter_XXXXXX have no axes, margins or colorbars.
Visualization and ProtoGate bilinear upsampling do not increase native feature resolution.
Edge supervision is positive-boundary continuous reliability weight, not a binary/soft target label.
The negative-ring loss remains unchanged and is not depicted as a separate candidate figure.
No target GT enters any probe. All four branches share the same fixed sample and spatial mask.
One compressed NPZ per 500-step node. With online_plot, intermediate ablations save one PNG, Full two.
Early inactive nodes are saved with states; online PNGs and raw arrays share one fixed probe.
manifest.snapshots records files, model/sample identity, completion/failure and elapsed time.
Final online-policy summaries group MPARC/PBCR; --iterations can select any existing saved nodes.
Source VI median/IQR summaries are static spectral evidence, not proof of independent pixel replicates.
Detailed CSV/prototype class counts are optional via mechanism_detailed_output=True, default off.
Basic losses, time/data_time, CUDA peak allocated MiB, validation and checkpoint protocols are unchanged.
Full images explain cooperation; single-branch images can explain isolated effects. Validation establishes performance.
'''
        path = reports / 'branch_evidence_mapping.md'
        if not path.exists():
            path.write_text(branch_mapping, encoding='utf-8')
        if self.cfg.get('online_plot', False):
            # Preserve old reports on resume while making the current policy explicit.
            current = reports / 'online_snapshot_policy.md'
            if not current.exists():
                current.write_text('# Current online snapshot policy\n'
                    'This supersedes older no-online-PNG descriptions in this analysis directory.\n'
                    'PNG: fixed_probe/snapshots/iter_XXXXXX_<kind>.png; one per intermediate group, two per Full.\n'
                    'Raw: fixed_probe/selected_arrays/iter_XXXXXX.npz; status: manifest.snapshots.\n'
                    'Online every 500 iterations, before validation; final summaries group MPARC/PBCR.\n\n'
                    + branch_mapping, encoding='utf-8')

    def load_sample(self):
        self.sample, description = load_probe_sample(self.probe_cfg, self.model)
        folder = self.root / 'fixed_probe'
        folder.mkdir(exist_ok=True)
        path = folder / 'sample_manifest.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != json.loads(json.dumps(description)):
            raise ValueError('Fixed probe sample/preprocessing changed in an existing analysis directory')
        if not path.exists():
            path.write_text(json.dumps(description, indent=2), encoding='utf-8')

    def consume(self):
        self.pending.extend(self.model.mechanism_records)
        self.model.mechanism_records.clear()
        completed = self.runner.iter + 1
        if completed % self.cfg['summary_interval'] == 0:
            self.summarize(completed)
        if self.cfg['probe_enabled'] and completed % self.cfg['probe_interval'] == 0:
            self.snapshot(completed)

    def summarize(self, completed):
        if self.cfg.get('output_mode', 'branches') == 'branches':
            return self.summarize_branches(completed)
        return self.summarize_detailed(completed)

    def summarize_branches(self, completed):
        samples = self.pending
        if not samples:
            return
        last = samples[-1]
        flags = last['modules']
        row = dict(iter=completed, experiment_name=self.manifest['experiment_name'],
            enabled_branches='+'.join(k for k in MODULES if flags[k]),
            data_source='training_batch_actual', n_sampled_batches=len(samples),
            sample_interval=self.cfg['sample_interval'], window_start=self.window_start)
        for name, fields in BRANCH_FIELDS.items():
            active = [s for s in samples if flags[name] and s['states'][name] in ('active', 'ramping')]
            if name == 'proto':
                active = [s for s in samples if s['proto_weight_active'] == 1]
            row[name + '_state'] = last['states'][name]
            row[name + '_sampled_batches'] = len(active)
            for field in fields:
                terms = [s['branch'][field] for s in active if field in s['branch']]
                denominator = sum(t[1] for t in terms)
                row[field] = sum(t[0] for t in terms) / denominator if denominator else 'NA'
            key = fields[0]
            row[name + '_valid_count'] = sum(s['branch'].get(key, (0,0))[1] for s in active)
        updates = [s for s in samples if flags['proto'] and s['proto_update_active'] == 1]
        counts = {key: sum(c[key] for s in updates for c in s['classes']) for key in
                  ('predicted_count', 'joint_selected_count', 'updated_count')}
        denominator = counts['predicted_count']
        row['proto_target_selected_ratio'] = counts['joint_selected_count']/denominator if denominator else 'NA'
        row['proto_target_update_ratio'] = counts['updated_count']/denominator if denominator else 'NA'
        row['proto_update_sampled_batches'] = len(updates)
        row['proto_selection_valid_count'] = denominator
        row['proto_selected_count'] = counts['joint_selected_count']
        row['proto_updated_count'] = counts['updated_count']
        row['tau_proto'] = last['tau_proto'] if flags['proto'] else 'NA'
        if getattr(self.model, 'phy_mode', None) == 'hierarchical_gv':
            row['head_version'] = 'hierarchical_gv'
            for key in ('r_veg', 'q_goji_given_veg', 's_phy_class_agreement', 'g_phy', 'attenuation'):
                name = 'hphy_' + key + '_mean'
                terms = [s['branch'][name] for s in samples if name in s['branch']]
                count = sum(term[1] for term in terms)
                row[name] = sum(term[0] for term in terms) / count if count else 'NA'
        append_unique_csv(self.root/'metrics/branch_mechanism_500.csv', [row], ['iter'])
        if self.cfg.get('detailed_output', False):
            self.summarize_detailed(completed)
        else:
            self.pending = []
            self.window_start = completed

    def summarize_detailed(self, completed):
        samples = self.pending
        if not samples:
            return
        count = sum(row['valid_count'] for row in samples)
        denominator = max(count, 1)
        last = samples[-1]
        row = dict(iter=completed, window_start=self.window_start,
            data_source='training_batch_actual', n_sampled_batches=len(samples),
            sample_interval=self.cfg['sample_interval'], valid_count=count,
            valid_area_ratio=count / max(sum(s['total_count'] for s in samples), 1),
            teacher_high_conf_ratio=sum(s['high_conf_count'] for s in samples) / denominator)
        for name in MEANS:
            row[name + '_mean_valid'] = (sum(s['sums'][name] for s in samples) / denominator
                if all(name in s['sums'] for s in samples) else '')
        for name, enabled in last.get('modules', {k: True for k in MODULES}).items():
            row[name + '_enabled'] = int(enabled)
            row[name + '_state'] = last.get('states', {}).get(name, 'active' if enabled else 'disabled')
        for key in ('tau_proto', 'tau_conf_proto', 'proto_valid_class_count',
                    'proto_update_active', 'proto_weight_active'):
            row[key] = last[key]
        updated = sum(c['updated_count'] for s in samples for c in s['classes'])
        predicted = sum(c['predicted_count'] for s in samples for c in s['classes'])
        row['proto_target_update_ratio'] = updated / max(predicted, 1) if last['classes'] else ''
        if last.get('modules', {}).get('phy', False):
            row.update(last.get('phy_parameters', {}))
            row['w_before_phy_mean_valid'] = row['w_base_mean_valid']
            for name in ('source_vegetation', 'source_nonvegetation', 'target_vegetation_pred',
                         'target_nonvegetation_pred', 'target_goji_pred', 'target_othervegetation_pred'):
                pairs = [s.get('phy_groups', {}).get(name, (0, 0)) for s in samples]
                n = sum(p[1] for p in pairs)
                row['r_phy_' + name + '_mean'] = sum(p[0] for p in pairs) / n if n else ''
                row['r_phy_' + name + '_count'] = n
            n = sum(s.get('phy_vegetation_count', 0) for s in samples)
            row['phy_downweighted_count'] = sum(s.get('phy_downweighted_count', 0) for s in samples)
            row['phy_vegetation_count'] = n
            row['phy_downweight_ratio'] = row['phy_downweighted_count'] / n if n else ''
            for name in ('loss_phy_src', 'loss_phy_tgt'):
                losses = [s.get('phy_losses', {}).get(name) for s in samples]
                row[name] = sum(losses) / len(losses) if all(v is not None for v in losses) else ''
            for group in ('pos', 'neg'):
                terms = [s.get('phy_losses', {}) for s in samples]
                n = sum(t.get(group + '_count', 0) for t in terms)
                row['phy_' + group + '_count'] = n
                row['loss_phy_' + group] = (sum(t.get(group + '_bce', 0) * t.get(group + '_count', 0)
                    for t in terms) / n if n else '')
        rows = []
        for cls in range(len(last['classes'])):
            item = dict(iter=completed, class_id=cls, class_name=last['classes'][cls]['class_name'],
                tau_proto=last['tau_proto'], tau_conf_proto=last['tau_conf_proto'],
                n_sampled_batches=len(samples), sample_interval=self.cfg['sample_interval'],
                data_source='training_batch_actual',
                update_active=last['proto_update_active'],
                prototype_initialized=last['classes'][cls]['prototype_initialized'])
            for key in ('predicted_count', 'confidence_pass_count', 'weight_pass_count',
                        'joint_selected_count', 'updated_count'):
                item[key] = sum(s['classes'][cls][key] for s in samples)
            item['selected_ratio'] = item['joint_selected_count'] / max(item['predicted_count'], 1)
            item['selected_ratio_all_valid'] = item['joint_selected_count'] / max(predicted, 1)
            rows.append(item)
        append_unique_csv(self.root / 'metrics/mechanism_500.csv', [row], ['iter'])
        if rows:
            append_unique_csv(self.root / 'metrics/proto_selection_500.csv', rows, ['iter', 'class_id'])
        self.pending = []
        self.window_start = completed

    def snapshot(self, completed):
        # Plotting/I/O are inside the RNG guard too; optional third-party
        # rendering must not consume randomness intended for the next batch.
        start = time.perf_counter()
        try:
            with readonly_probe(self.model):
                self._snapshot(completed)
                for probe in getattr(self, 'extra_probes', ()):
                    probe.snapshot(completed)
        except Exception as error:
            if self.cfg.get('output_mode', 'branches') != 'branches':
                raise
            self.runner.logger.error('Mechanism snapshot FAILED iter=%d: %s',
                                     completed, error, exc_info=True)
            self.record_snapshot(completed, snapshot_status='failed', error=str(error),
                mechanism_probe_time=time.perf_counter()-start,
                traceback=traceback.format_exc(), figures=[p.relative_to(self.root).as_posix()
                    for p in (self.root/'fixed_probe/snapshots').glob('iter_{:06d}_*.png'.format(completed))])
            if self.cfg.get('strict', True):
                raise

    def _snapshot(self, completed):
        if self.cfg.get('output_mode', 'branches') == 'branches':
            return self.snapshot_branches(completed)
        start = time.perf_counter()
        device = next(self.model.parameters()).device
        before = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
        image, vi, metas = self.sample
        image, vi = image.to(device), vi.to(device)
        valid = torch.ones((1, 1, *image.shape[-2:]), device=device)
        top, bottom, left, right = self.probe_cfg['margins']
        if top: valid[:, :, :top] = 0
        if bottom: valid[:, :, -bottom:] = 0
        if left: valid[:, :, :, :left] = 0
        if right: valid[:, :, :, -right:] = 0
        folder = self.root / 'fixed_probe'
        arrays = folder / 'arrays'
        arrays.mkdir(exist_ok=True)
        path = arrays / ('iter_{:06d}.npz'.format(completed))
        reused = path.exists()
        state = model_digest(self.model)
        metadata = dict(modules=module_flags(self.model), states=module_states(self.model),
            iter=completed, sample_id=Path(metas[0]['filename']).name,
            input_view=self.probe_cfg.get('channel_order', [])[:3],
            config_sha256=self.identity, state_sha256=state, data_source='fixed_probe_recomputed',
            precision='cuda_autocast' if device.type == 'cuda' else 'cpu_float32',
            sample_tensor_sha256=json.loads((folder / 'sample_manifest.json').read_text())['tensor_sha256'])
        if metadata['modules']['edge']:
            metadata['edge_pair_prior_ready'] = bool(self.model.struct_edge_pair_prior_ready)
        compact = getattr(self, 'physiology', self.cfg.get('physiology', False))
        metadata['physiology_compact'] = compact
        if metadata['modules']['phy']:
            metadata['head_version'] = getattr(self.model, 'phy_mode', 'legacy_binary')
        generic = not compact or any(metadata['modules'][k] for k in ('struct', 'proto', 'edge'))
        phy_path = self.root / 'physiology/arrays' / ('iter_{:06d}_phy.npz'.format(completed))
        required = ([path] if generic else []) + ([phy_path] if compact else [])
        existing = {}
        for saved_path in required:
            if saved_path.exists():
                with np.load(saved_path, allow_pickle=False) as saved:
                    if json.loads(str(saved['metadata'])) != metadata:
                        raise ValueError('Existing probe node belongs to a different model state: ' + str(saved_path))
                    existing.update({key: saved[key] for key in saved.files})
        reused = all(p.exists() for p in required)
        if reused:
            maps = existing
        else:
            maps = probe_arrays(self.model, image, vi, metas, valid)
            norm = metas[0]['img_norm_cfg']
            view = image[0, :3].detach().float().cpu().numpy().transpose(1, 2, 0)
            maps['target_image'] = np.clip((view * np.asarray(norm['std'][:3]) +
                                          np.asarray(norm['mean'][:3])) / 255.0, 0, 1).astype(np.float32)
            maps['metadata'] = np.array(json.dumps(metadata, sort_keys=True))
            if self.cfg['save_raw']:
                payloads = []
                if generic:
                    omitted = {'r_phy', 'm_phy', 'w_t_phy', 'delta_w_phy'} if compact else set()
                    payloads.append((path, {k: v for k, v in maps.items() if k not in omitted}))
                if compact:
                    phy = {k: maps[k] for k in ('r_phy', 'm_phy', 'w_t_phy', 'delta_w_phy',
                           'valid_mask', 'target_image', 'metadata', 'rule_iter')}
                    phy['w_before_phy'] = maps['w_base']
                    if 'r_veg' in maps:
                        phy.update({k: v for k, v in maps.items() if k not in
                                    ('target_image', 'metadata')})
                        phy['w_after_phy'] = maps['w_t_phy']
                    payloads.append((phy_path, phy))
                for saved_path, payload in payloads:
                    if saved_path.exists():
                        continue
                    saved_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = saved_path.with_suffix('.npz.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, **payload)
                    temporary.replace(saved_path)
        if self.cfg['make_figures'] and generic:
            from tools.build_doris_mechanism_figures import render_node
            render_node(maps, folder / 'snapshots', completed, overwrite=False)
        elapsed = time.perf_counter() - start
        after = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
        new_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
        if not reused:
            append_unique_csv(self.root / 'metrics/probe_cost.csv', [dict(
                iter=completed, mechanism_probe_time=elapsed, memory_before_mib=before / 2**20,
                memory_after_mib=after / 2**20, run_peak_increase_mib=max(new_peak - peak, 0) / 2**20)], ['iter'])
        self.runner.logger.info('Mechanism fixed probe iter=%d: %.3fs (including I/O and rendering)',
                                completed, elapsed)

    def snapshot_branches(self, completed):
        start = time.perf_counter()
        device = next(self.model.parameters()).device
        before = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
        image, vi, metas = self.sample
        image, vi = image.to(device), vi.to(device)
        valid = torch.ones((1,1,*image.shape[-2:]), device=device)
        top, bottom, left, right = self.probe_cfg['margins']
        if top: valid[:,:,:top] = 0
        if bottom: valid[:,:,-bottom:] = 0
        if left: valid[:,:,:,:left] = 0
        if right: valid[:,:,:,-right:] = 0
        sample = json.loads((self.root/'fixed_probe/sample_manifest.json').read_text())
        metadata = dict(iter=completed, rule_iter=self.model.local_iter,
            sample_id=Path(metas[0]['filename']).name, sample_tensor_sha256=sample['tensor_sha256'],
            config_sha256=self.identity, state_sha256=model_digest(self.model),
            modules=module_flags(self.model), states=module_states(self.model),
            data_source='fixed_probe_recomputed', output_mode='branches',
            input_view=self.probe_cfg.get('channel_order', [])[:3])
        metadata.update(capture_mode='online_after_train_iter', checkpoint=None,
            git_commit=self.manifest.get('git_commit'),
            target_path=sample.get('target_path', metas[0]['filename']),
            direction=self.probe_cfg.get('direction'),
            original_shape=sample.get('original_shape', list(image.shape[-2:])),
            input_shape=list(image.shape[-2:]), crop=None, padding=[0,0,0,0],
            spatial_mapping='full native input grid' if self.probe_cfg.get('filename') else 'legacy deterministic resize')
        if metadata['modules']['proto']:
            gate = self.model.proto_gate
            metadata['prototype'] = dict(target_update_start=gate.target_update_start,
                weight_start=gate.weight_start, loss_start=gate.loss_start,
                target_update_active=bool(gate.target_update and self.model.local_iter >= gate.target_update_start),
                valid_classes=int(gate.valid_proto.sum()), target_valid_classes=int(gate.target_valid_proto.sum()),
                valid_class_ids=gate.valid_proto.nonzero().flatten().tolist(),
                target_valid_class_ids=gate.target_valid_proto.nonzero().flatten().tolist(),
                tau_proto=gate.tau_proto, tau_conf_proto=gate.tau_conf_proto, min_pixels=gate.min_pixels)
        metadata['experiment_name'] = self.manifest['experiment_name']
        if metadata['modules']['edge']:
            metadata['edge_supervision_max'] = float(self.model.struct_edge_loss_weight_tgt)
        path = self.root/'fixed_probe/selected_arrays'/('iter_{:06d}.npz'.format(completed))
        reused = path.exists()
        previous = self.manifest.get('snapshots', {}).get(str(completed), {})
        if reused and previous.get('snapshot_status') == 'completed':
            expected_files = [self.root / p for p in previous.get('figures', [])]
            if previous.get('online_plot') == self.cfg.get('online_plot', False) and all(
                    p.is_file() and p.stat().st_size for p in [path] + expected_files):
                # Validate model/config below, but do not rewrite valid historical records.
                completed_before = True
            else:
                completed_before = False
        else:
            completed_before = False
        if reused:
            with np.load(path, allow_pickle=False) as saved:
                saved_meta = json.loads(str(saved['metadata']))
                identity_fields = ('iter','rule_iter','sample_id','sample_tensor_sha256',
                                   'config_sha256','state_sha256','modules','states','data_source')
                if any(saved_meta.get(k) != metadata.get(k) for k in identity_fields):
                    raise ValueError('Existing branch node differs in model/sample/config: ' + str(path))
                payload = {key: saved[key] for key in saved.files}
        if completed_before:
            if self.cfg.get('online_plot', False):
                from tools.build_doris_mechanism_figures import render_online_node
                render_online_node(payload, self.root/'fixed_probe/snapshots', completed,
                                   array_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            return
        self.record_snapshot(completed, snapshot_status='in_progress', sample_id=metadata['sample_id'],
            capture_mode=metadata['capture_mode'], target_path=metadata['target_path'],
            direction=metadata['direction'], original_shape=metadata['original_shape'],
            config_sha256=self.identity, state_sha256=metadata['state_sha256'],
            sample_tensor_sha256=metadata['sample_tensor_sha256'],
            enabled_branches=metadata['modules'], states=metadata['states'],
            array_path=path.relative_to(self.root).as_posix(), figures=[],
            online_plot=self.cfg.get('online_plot', False))
        if not reused and self.cfg['save_raw']:
            payload = branch_probe_payload(self.model, image, vi, metas, valid, metadata)
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,**payload)
            temporary.replace(path)
        figures = []
        if self.cfg.get('online_plot', False):
            from tools.build_doris_mechanism_figures import render_online_node
            figures = render_online_node(payload, self.root/'fixed_probe/snapshots', completed,
                array_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if len(figures) != (2 if all(metadata['modules'].values()) else 1):
                raise IOError('Online renderer did not return all required PNGs')
        required = ([path] if self.cfg['save_raw'] else []) + figures
        if not required or not all(p.is_file() and p.stat().st_size for p in required):
            raise IOError('Mechanism snapshot files are missing or empty at iter {}'.format(completed))
        cost_path = self.root/'metrics/probe_cost.csv'
        has_cost = False
        if cost_path.exists():
            with cost_path.open(newline='', encoding='utf-8') as stream:
                has_cost = any(int(row['iter']) == completed for row in csv.DictReader(stream))
        if not has_cost:
            after = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
            new_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
            append_unique_csv(cost_path, [dict(iter=completed,
                mechanism_probe_time=time.perf_counter()-start, memory_before_mib=before/2**20,
                memory_after_mib=after/2**20, run_peak_increase_mib=max(new_peak-peak,0)/2**20)], ['iter'])
        elapsed = time.perf_counter() - start
        self.record_snapshot(completed, snapshot_status='completed',
            figures=[p.relative_to(self.root).as_posix() for p in figures],
            spatial_maps=('fixed_probe/snapshots/spatial/iter_{:06d}'.format(completed)
                          if self.cfg.get('online_plot', False) else None),
            native_panels={name: 'fixed_probe/snapshots/{}/iter_{:06d}'.format(name, completed)
                           for name in ('full256', 'center196') if (self.root /
                               'fixed_probe/snapshots' / name / 'iter_{:06d}'.format(completed)).is_dir()},
            mechanism_probe_time=elapsed, array_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.runner.logger.info('Mechanism snapshot completed iter=%d: %.3fs; NPZ=%s; PNG=%s',
                                completed, elapsed, str(path), ', '.join(str(p) for p in figures))

    def finish(self):
        if self.cfg['probe_enabled'] and self.cfg['make_figures']:
            from tools.build_doris_mechanism_figures import build_figures
            start = time.perf_counter()
            with readonly_probe(self.model):
                if self.output_mode == 'branches':
                    if self.modules['phy']:
                        from tools.build_doris_physiology_figures import analyze_vi
                        analyze_vi(self.source_cfg, self.root/'physiology', make_figure=False)
                    output = build_figures(self.root)
                    self.runner.logger.info('Core branch evidence: %s', str(output))
                    return
                if not self.physiology or any(self.modules[k] for k in ('struct', 'proto', 'edge')):
                    build_figures(self.root)
                if self.physiology:
                    from tools.build_doris_physiology_figures import analyze_vi, build_phy_response_figure
                    analyze_vi(self.source_cfg, self.root / 'physiology')
                    build_phy_response_figure(self.root)
            self.runner.logger.info('Mechanism candidate figures: %s (%.3fs, saved-data/source-VI analysis)',
                                    str(self.root), time.perf_counter() - start)
