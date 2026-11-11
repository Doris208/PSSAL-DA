"""NumPy-only Structural diagnostics shared by online probes and saved replays."""
import json
import numpy as np

STRUCT_CLASSES = (2, 3, 4)
RESPONSES = ('R_conf', 'R_loc', 'R_adj', 'R_prob', 'R_struct')
CONTRIBUTIONS = ('C_loc', 'C_adj', 'C_prob')
ALIASES = dict(image_rgb='target_image', teacher_pseudo_label='pseudo_label',
    R_conf='r_conf', R_loc='r_loc', R_adj='r_adj', R_prob='r_prob', R_struct='r_struct',
    W_before='w_before_struct', W_after='w_after_struct')


def relative_struct_attenuation(before, after, label, valid):
    eligible = np.asarray(valid, dtype=bool) & np.isin(label, STRUCT_CLASSES) & (before > 0)
    attenuation = np.zeros(np.asarray(before).shape, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        attenuation[eligible] = 1-after[eligible].astype(np.float64)/before[eligible]
    return attenuation, eligible


def prepare_struct_evidence(saved, metadata=None, config=None):
    """Derive display arrays from captured weights, with explicit validity masks."""
    meta = dict(metadata or json.loads(str(saved['metadata'])))
    cfg = dict(config or meta.get('struct_cfg') or meta.get('gate') or {})
    if cfg.get('fusion_mode') != 'doris_confidence_guided':
        raise ValueError('Structural diagnostics require the saved confidence-guided struct_cfg')
    if list(cfg.get('apply_to_classes', cfg.get('struct_positive_classes', STRUCT_CLASSES))) != list(STRUCT_CLASSES):
        raise ValueError('Expected Structural classes Building/Water/Road with IDs 2/3/4')
    for key in ('continuity_weight', 'adjacency_weight', 'semantic_weight', 'struct_min_weight',
                'struct_filter_threshold', 'struct_filter_mode', 'struct_warmup_iters', 'struct_rampup_iters'):
        if key not in cfg:
            raise ValueError('Missing saved structural configuration: ' + key)
    arrays = {key: np.asarray(value).copy() for key, value in saved.items()
              if np.asarray(value).ndim >= 2}
    for destination, source in ALIASES.items():
        if destination not in arrays and source in arrays:
            arrays[destination] = arrays[source].copy()
    required = {'image_rgb', 'teacher_pseudo_label', 'valid_mask', 'W_before', 'W_after',
                'adjacency_available', 'semantic_available', *RESPONSES}
    if required-set(arrays):
        raise ValueError('Missing real Structural inputs: '+', '.join(sorted(required-set(arrays))))
    for key in required:
        if arrays[key].shape[:2] != (256, 256):
            raise ValueError('Structural diagnostics require the native 256 grid: '+key)
    if 'struct_prediction' in arrays:
        prediction = arrays['struct_prediction']
        if prediction.shape != (256, 256) or prediction.dtype.kind not in 'iu' or not np.isin(prediction, [0, 1, 2, 3, 4]).all():
            raise ValueError('struct_prediction must contain native-grid Student class IDs 0..4')
    if arrays['image_rgb'].dtype != np.uint8:
        rgb = arrays['image_rgb']
        if rgb.shape != (256, 256, 3) or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError('Expected uint8 RGB or normalized target_image in [0,1]')
        arrays['image_rgb'] = (rgb*255).round().astype(np.uint8)
    arrays['target_rgb'] = arrays['image_rgb'].copy()
    valid = arrays['valid_mask'].astype(bool)
    label = arrays['teacher_pseudo_label']
    if not np.isin(label, [0, 1, 2, 3, 4, 255]).all():
        raise ValueError('Unexpected Teacher reduced class IDs')
    scope = valid & np.isin(label, STRUCT_CLASSES)
    before, after = arrays['W_before'], arrays['W_after']
    attenuation, eligible = relative_struct_attenuation(before, after, label, valid)
    arrays.update(A_struct=attenuation, A_struct_valid_mask=eligible,
        struct_class_mask=scope, teacher_struct_mask=scope, struct_display_mask=scope.copy(), teacher_pred=label.copy(),
        positive_struct_weight_mask=eligible, positive_base_weight_mask=valid & (before > 0),
        building_mask=valid & (label == 2), water_mask=valid & (label == 3), road_mask=valid & (label == 4))
    arrays.setdefault('teacher_confidence', arrays['R_conf'].copy())
    crop = np.zeros((256, 256), dtype=bool)
    crop[30:226, 30:226] = True
    arrays['crop_mask_196'] = crop
    arrays['adjacency_available'] = arrays['adjacency_available'].astype(bool) & cfg.get('use_source_adjacency_ema', True)
    arrays['semantic_available'] = arrays['semantic_available'].astype(bool) & cfg.get('use_semantic_proto', True)
    label_valid = (label >= 0) & (label < 5)
    denominator = np.zeros((256, 256), dtype=np.float64)
    weighted = []
    for coefficient, key, available in (
            (cfg['continuity_weight'], 'R_loc', label_valid),
            (cfg['adjacency_weight'], 'R_adj', arrays['adjacency_available'] & label_valid),
            (cfg['semantic_weight'], 'R_prob', arrays['semantic_available'] & label_valid)):
        usable = available & np.isfinite(arrays[key])
        denominator += coefficient*usable
        weighted.append(coefficient*np.where(usable, arrays[key], 0))
    denominator = np.maximum(denominator, cfg.get('eps', 1e-6))
    for key, numerator in zip(CONTRIBUTIONS, weighted):
        arrays[key] = arrays['R_conf']*numerator/denominator
    completed = int(meta.get('rule_iter', cfg.get('rule_iter', meta.get('iter', 0))))
    ramp = min(1., max(0., (completed-cfg['struct_warmup_iters'])/max(float(cfg['struct_rampup_iters']), 1.)))
    if not cfg.get('use_target_gate', True):
        ramp = 0.
    raw_gate = np.maximum(arrays['R_struct'], cfg['struct_min_weight'])
    below = arrays['R_struct'] < cfg['struct_filter_threshold']
    raw_gate = np.where(below, 0 if cfg['struct_filter_mode'] == 'hard' else raw_gate*.5, raw_gate)
    expected_gate = np.ones((256, 256))
    expected_gate[scope] = (1-ramp)+ramp*raw_gate[scope]
    floor = cfg['struct_min_weight']
    if cfg['struct_filter_threshold'] > 0:
        floor = 0 if cfg['struct_filter_mode'] == 'hard' else floor*.5
    theoretical_max = ramp*(1-floor)
    tolerance = 1e-5
    precision = meta.get('struct_probability_dtype', meta.get('checkpoint_struct', {}).get('struct_probability_dtype'))
    amp = precision == 'float16' or 'cuda' in str(meta.get('precision', '')).lower()
    # Old online nodes do not record AMP. Record the tolerance used for those nodes.
    fusion_tolerance = max(tolerance, float(np.finfo(np.float16).eps)) if amp or precision is None else tolerance
    errors = []
    for name, values in (('W_before', before), ('W_after', after), *[(key, arrays[key]) for key in RESPONSES]):
        if not np.isfinite(values[valid]).all():
            errors.append('Non-finite '+name)
    for key in RESPONSES:
        if np.any(arrays[key][valid] < 0) or np.any(arrays[key][valid] > 1):
            errors.append('Response outside [0,1]: '+key)
    if np.any(before[valid] < 0) or np.any(after[valid] < 0):
        errors.append('Negative supervision weight')
    if np.any(after[valid] > before[valid]+tolerance):
        errors.append('Supervision weight increased')
    if not np.isfinite(attenuation[eligible]).all():
        errors.append('Non-finite relative attenuation')
    fusion_error = np.abs(sum(arrays[key] for key in CONTRIBUTIONS)-arrays['R_struct'])[valid]
    gate_error = np.abs(after-before*expected_gate)[valid]
    identity_error = np.abs(after[eligible]-before[eligible]*(1-attenuation[eligible]))
    max_fusion = float(fusion_error.max()) if fusion_error.size else 0.
    max_gate = float(gate_error.max()) if gate_error.size else 0.
    max_identity = float(identity_error.max()) if identity_error.size else 0.
    if not np.isfinite(max_fusion) or max_fusion > fusion_tolerance:
        errors.append('Contribution sum differs from captured R_struct')
    if not np.isfinite(max_gate) or max_gate > tolerance:
        errors.append('Captured weights differ from scheduled Structural gate')
    if not np.isfinite(max_identity) or max_identity > tolerance:
        errors.append('W_after = W_before * (1-A_struct) failed')
    if np.any(attenuation[eligible] < -tolerance) or np.any(attenuation[eligible] > theoretical_max+tolerance):
        errors.append('Attenuation outside the saved gate schedule bounds')
    validation = dict(passed=not errors, errors=errors,
        fusion_max_abs_error=max_fusion if np.isfinite(max_fusion) else None,
        gate_max_abs_error=max_gate if np.isfinite(max_gate) else None,
        attenuation_identity_max_abs_error=max_identity if np.isfinite(max_identity) else None,
        weight_tolerance=tolerance, fusion_tolerance=fusion_tolerance,
        theoretical_attenuation_range=[0., theoretical_max])
    meta.update(struct_cfg=cfg, rule_iter=completed, structural_visualization_version=4,
        panel_sources={**{key: 'EMA Teacher probabilities and labels with saved source priors' for key in RESPONSES},
            'struct_prediction': 'Student test-mode inference from the selected model state'},
        attenuation_definition='1 - W_after/W_before on valid Teacher classes 2/3/4 with W_before > 0',
        attenuation_invalid_storage='zero with independent A_struct_valid_mask',
        attenuation_color_range=[0., theoretical_max if theoretical_max > 0 else 1.],
        attenuation_warmup_neutral=theoretical_max == 0, struct_validation=validation)
    return arrays, meta


def values_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(count=0, mean=None, std=None, min=None, max=None, p10=None, p90=None)
    return dict(count=int(values.size), mean=float(values.mean()), std=float(values.std()),
        min=float(values.min()), max=float(values.max()), p10=float(np.quantile(values, .1)),
        p90=float(np.quantile(values, .9)))


def struct_statistics(arrays, meta):
    """One-image descriptions. Ratios use all valid center196 pixels."""
    a = {key: value[30:226, 30:226] for key, value in arrays.items()}
    valid, labels = a['valid_mask'].astype(bool), a['teacher_pseudo_label']
    scope = a['struct_class_mask']
    counts = int(valid.sum())
    class_rows = []
    for name, mask in [('Building', a['building_mask']), ('Water', a['water_mask']),
                       ('Road', a['road_mask']), ('NonVeg', scope)]:
        active = mask & a['A_struct_valid_mask']
        attenuation = values_summary(a['A_struct'][active])
        row = dict(class_name=name, class_pixel_count=int(mask.sum()),
            struct_active_pixel_count=int(active.sum()),
            struct_active_pixel_ratio=float(active.sum()/counts) if counts else None,
            struct_changed_pixel_count=int((active & (a['A_struct'] > 1e-7)).sum()),
            W_before_struct_mean=values_summary(a['W_before'][active])['mean'],
            W_after_struct_mean=values_summary(a['W_after'][active])['mean'])
        row.update({'A_struct_'+key: attenuation[key] for key in ('mean', 'std', 'p10', 'p90', 'max')})
        for region, region_mask in (('interior', a['R_loc'] == 1), ('mixed_neighborhood', a['R_loc'] < 1)):
            regional = values_summary(a['A_struct'][active & region_mask])
            row[region+'_pixel_count'] = regional['count']
            row[region+'_A_struct_mean'] = regional['mean']
        class_rows.append(row)
    responses = {}
    for key in RESPONSES+('A_struct',)+CONTRIBUTIONS:
        mask = scope.copy()
        if key == 'A_struct':
            mask &= a['A_struct_valid_mask']
        if key in ('R_adj', 'R_prob'):
            mask &= a['adjacency_available' if key == 'R_adj' else 'semantic_available']
        responses[key] = values_summary(a[key][mask])
    correlations = {}
    for left, right in (('R_loc', 'R_adj'), ('R_loc', 'R_prob'), ('R_adj', 'R_prob'), ('R_conf', 'R_struct')):
        mask = scope.copy()
        for key in (left, right):
            if key in ('R_adj', 'R_prob'):
                mask &= a['adjacency_available' if key == 'R_adj' else 'semantic_available']
        x, y = a[left][mask], a[right][mask]
        correlations[left+'__'+right] = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and np.ptp(x) > 0 and np.ptp(y) > 0 else None
    contributions = {key: responses[key]['mean'] for key in CONTRIBUTIONS}
    total = sum(value or 0 for value in contributions.values())
    ratios = {key: value/total if value is not None and total > 0 else None for key, value in contributions.items()}
    ratio = float(scope.sum()/counts) if counts else 0.
    corr = correlations['R_loc__R_adj']
    score = .5*ratio+.3*(responses['R_struct']['std'] or 0)+.2*(1-corr) if corr is not None else None
    spatial = {}
    for name, mask in (('interior', scope & (a['R_loc'] == 1)), ('mixed_neighborhood', scope & (a['R_loc'] < 1))):
        spatial[name] = values_summary(a['A_struct'][mask & a['A_struct_valid_mask']])
    return dict(sample_id=meta.get('sample_id'), iter=meta.get('rule_iter'), scope='center196',
        valid_pixel_count=counts, struct_class_ratio=ratio, attenuation_by_class=class_rows,
        class_distribution=[dict(class_id=cls, class_name=name, count=int((valid & (labels == cls)).sum()),
            ratio=float((valid & (labels == cls)).sum()/counts) if counts else None)
            for cls, name in enumerate(('Goji', 'Vegetation', 'Building', 'Water', 'Road'))],
        responses=responses, correlations=correlations, contribution_means=contributions,
        contribution_ratios=ratios, attenuation_by_neighborhood=spatial, selection_score=score,
        selection_score_definition='0.5*struct_class_ratio + 0.3*std(R_struct) + 0.2*(1-corr(R_loc,R_adj)); descriptive sample selection only',
        struct_active_pixel_ratio_denominator='all valid center196 pixels',
        standard_deviation_definition='population standard deviation over applicable pixels (ddof=0)',
        struct_active_definition='valid Teacher structural classes with W_before > 0, including neutral warmup',
        neighborhood_definition='interior has R_loc = 1, mixed_neighborhood has R_loc < 1, both defined from Teacher predictions',
        validation=meta['struct_validation'])
