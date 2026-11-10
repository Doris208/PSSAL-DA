"""Reference labels and class summaries for offline physiology interpretation."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

RESPONSE_KEYS = ('w_conf', 'r_phy', 'w_before_phy', 'w_after_phy', 'delta_w_phy')
INTERPRETATION = 'VI-driven response and pseudo-supervision reweighting, evaluated for Goji/other-vegetation confusion'


def reference_path(config, filename):
    validation = config['data']['val']
    return (Path(validation['data_root']) / validation['ann_dir'] / filename).resolve()


def add_reference_evidence(payload, reference, annotation):
    """Summarize saved responses using existing validation labels after inference."""
    valid = payload['valid_mask'].astype(bool)
    class_names = annotation['class_names']
    allowed = list(range(len(class_names))) + [255]
    if reference.shape != valid.shape or not np.isin(reference, allowed).all():
        raise ValueError('Reference labels differ from the original grid or evaluation mapping')
    for key in RESPONSE_KEYS:
        if key not in payload or payload[key].shape != reference.shape:
            raise ValueError('Missing or unaligned physiology response: ' + key)
    rows = []
    for class_id, name in enumerate(class_names):
        mask = valid & (reference == class_id)
        count = int(mask.sum())
        row = dict(class_id=class_id, class_name=name, valid_pixels=count)
        for key in RESPONSE_KEYS:
            values = payload[key][mask]
            row[key + '_mean'] = float(values.mean()) if count else None
            row[key + '_std'] = float(values.std()) if count else None
        row['teacher_correct_fraction'] = float((payload['pseudo_label'][mask] == class_id).mean()) if count else None
        row['downweighted_fraction'] = float((payload['delta_w_phy'][mask] < 0).mean()) if count else None
        rows.append(row)
    metadata = json.loads(str(payload['metadata']))
    metadata['phy_evidence'] = dict(annotation, interpretation=INTERPRETATION,
        response_supervision=('Source vegetation BCE plus source-only balanced Goji/Vegetation BCE'
                              if 'r_veg' in payload else
                              'Goji and other vegetation share the positive label of the physiology head'),
        annotation_usage='Offline reference and class summaries after model forward',
        statistics_scope='Valid pixels grouped by existing validation ground truth',
        ignore_index=255, classes=rows)
    if 'r_veg' in payload:
        metadata['phy_evidence']['hierarchical_reliability'] = hphy_reliability_statistics([dict(payload, reference_label=reference)])
    payload['reference_label'] = reference.copy()
    payload['metadata'] = np.array(json.dumps(metadata, sort_keys=True))
    return payload


def load_reference_annotation(config, filename):
    from mmseg.datasets.pipelines import LoadAnnotations
    from mmseg.datasets.goji_meta import GOJI_CLASSES, GOJI_PALETTE

    validation = config['data']['val']
    if validation['type'] != 'GojiDataset' or validation.get('classes') is not None:
        raise ValueError('Reference evidence requires the standard GojiDataset class mapping')
    path = reference_path(config, filename)
    if not path.is_file():
        raise FileNotFoundError('Validation annotation required for physiology evidence: ' + str(path))
    result = LoadAnnotations(reduce_zero_label=True)(dict(
        ann_info=dict(seg_map=str(path)), seg_fields=[]))
    annotation = dict(annotation_path=str(path),
        annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        label_mapping='Existing LoadAnnotations(reduce_zero_label=True)',
        class_names=list(GOJI_CLASSES), palette=GOJI_PALETTE)
    return result['gt_semantic_seg'], annotation


def load_phy_reference(payload, config):
    metadata = json.loads(str(payload['metadata']))
    reference, annotation = load_reference_annotation(config,metadata['sample_id'])
    return add_reference_evidence(payload, reference, annotation)


def class_statistics_csv(evidence):
    fields = ['class_id', 'class_name', 'valid_pixels',
              'teacher_correct_fraction', 'downweighted_fraction']
    fields += [key + '_' + statistic for key in RESPONSE_KEYS for statistic in ('mean', 'std')]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(evidence['classes'])
    return stream.getvalue()


def error_auc(errors, scores):
    """Tie-aware ROC AUC and average precision, with explicit missing classes."""
    errors = np.asarray(errors, dtype=bool)
    positives, negatives = int(errors.sum()), int((~errors).sum())
    if not positives or not negatives:
        return dict(roc_auc=None, pr_auc=None, errors=positives, correct=negatives)
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(order) - 1]
    tp = np.cumsum(errors[order])[ends].astype(float)
    fp = (ends + 1) - tp
    recall = np.r_[0.0, tp / positives]
    false_positive = np.r_[0.0, fp / negatives]
    return dict(roc_auc=float(np.trapz(recall, false_positive)),
                pr_auc=float(np.sum(np.diff(recall) * tp / (tp + fp))),
                errors=positives, correct=negatives)


def matched_budget_retention(errors, scores, budget, base_weights=None):
    """Retain the lowest error scores at equal absolute pseudo-weight mass."""
    if len(errors) == 0:
        return dict(correct_retained=None, error_reduction=None)
    order = np.argsort(scores, kind='stable')
    base_weights = np.ones(len(order)) if base_weights is None else base_weights
    cumulative_mass = np.r_[0., np.cumsum(base_weights[order])]
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(order)]
    weights = np.zeros(len(order), dtype=float)
    for start, end in zip(starts, ends):
        weights[order[start:end]] = np.clip(
            (budget - cumulative_mass[start]) /
            (cumulative_mass[end] - cumulative_mass[start]), 0, 1)
    correct = ~errors
    return dict(correct_retained=float(weights[correct].mean()) if correct.any() else None,
                error_reduction=float((1 - weights[errors]).mean()) if errors.any() else None)


def hphy_reliability_statistics(payloads, bootstrap_samples=200, seed=0):
    """Offline multi-tile statistics. Every payload must already contain reference GT."""
    if not payloads:
        raise ValueError('At least one independently evaluated tile is required')
    keys = ('r_veg', 'q_goji_given_veg', 's_phy_class_agreement', 'g_phy', 'attenuation_phy')
    groups = []
    for truth, predicted in ((0, 0), (0, 1), (1, 1), (1, 0)):
        row = dict(gt=truth, teacher=predicted)
        count = 0
        for key in keys:
            values = []
            for payload in payloads:
                mask = (payload['valid_mask'].astype(bool) & (payload['reference_label'] == truth)
                        & (payload['pseudo_label'] == predicted) & (payload['w_before_phy'] > 0))
                values.append(payload[key][mask])
            values = np.concatenate(values)
            count = len(values)
            row[key] = (dict(mean=float(values.mean()), median=float(np.median(values)),
                             quantiles=np.quantile(values, [.05, .25, .75, .95]).tolist())
                        if count else None)
        row['valid_pixels'] = count
        groups.append(row)
    rows = []
    rng = np.random.RandomState(seed)
    for predicted in (0, 1):
        for high_only in (False, True):
            tiles = []
            for payload in payloads:
                reference = payload['reference_label']
                mask = (payload['valid_mask'].astype(bool) & (payload['pseudo_label'] == predicted)
                        & (reference >= 0) & (reference < 5) & (payload['w_before_phy'] > 0))
                if high_only:
                    mask &= payload['w_conf'] >= .968
                tiles.append((reference[mask] != predicted, 1 - payload['g_phy'][mask],
                              1 - payload['w_conf'][mask], payload['w_before_phy'][mask]))
            errors, attenuation, uncertainty, base = [np.concatenate([tile[i] for tile in tiles]) for i in range(4)]
            row = dict(teacher=predicted, high_confidence_only=high_only,
                       valid_pixels=len(errors), error_prevalence=float(errors.mean()) if len(errors) else None)
            for name, scores in (('hphy', attenuation), ('teacher_uncertainty', uncertainty)):
                metrics = error_auc(errors, scores)
                intervals = []
                if len(tiles) >= 2:
                    for _ in range(bootstrap_samples):
                        sampled = rng.randint(len(tiles), size=len(tiles))
                        sampled_errors = np.concatenate([tiles[i][0] for i in sampled])
                        sampled_scores = np.concatenate([tiles[i][1 if name == 'hphy' else 2] for i in sampled])
                        auc = error_auc(sampled_errors, sampled_scores)
                        if auc['roc_auc'] is not None:
                            intervals.append([auc['roc_auc'], auc['pr_auc']])
                metrics['tile_bootstrap_95_ci'] = (
                    np.quantile(intervals, [.025, .975], axis=0).tolist() if intervals else None)
                metrics['bootstrap_valid_replicates'] = len(intervals)
                metrics['matched_weight_budget'] = matched_budget_retention(
                    errors, scores, float((base * (1 - attenuation)).sum()), base)
                row[name] = metrics
            correct = ~errors
            row['actual_hphy_weight_retention'] = dict(
                correct_retained=float((1 - attenuation)[correct].mean()) if correct.any() else None,
                error_reduction=float(attenuation[errors].mean()) if errors.any() else None)
            rows.append(row)
    return dict(tiles=len(payloads), groups=groups, error_detection=rows,
                score_direction='Larger attenuation or Teacher uncertainty predicts an error',
                scope='Valid reference pixels with positive pre-phy weights; errors include all five GT classes',
                confidence_intervals='Tile bootstrap; unavailable for a single tile or absent error/correct group',
                budget_definition='Equal sum of absolute post-gate pseudo weights; retention rates count pixels',
                pr_auc_definition='Average precision using complete score-tie groups', seed=seed)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Offline H-Phy reliability statistics from reference-enriched NPZ files')
    parser.add_argument('--arrays', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payloads, identities = [], set()
    for path in args.arrays:
        with np.load(path, allow_pickle=False) as saved:
            payload = {key: saved[key] for key in saved.files}
        metadata = json.loads(str(payload['metadata']))
        if metadata.get('head_version') != 'hierarchical_gv' or any(
                metadata.get(key) is None for key in ('sample_id', 'iter', 'config_sha256')):
            raise ValueError('Offline H-Phy inputs require head version, sample, iteration and config identity')
        identity = (metadata.get('sample_id'), metadata.get('iter'))
        if identity in identities:
            raise ValueError('Duplicate sample/iteration in offline population')
        identities.add(identity)
        payloads.append(payload)
    protocols = {(json.loads(str(p['metadata'])).get('config_sha256'),
                  json.loads(str(p['metadata'])).get('iter')) for p in payloads}
    if len(protocols) != 1:
        raise ValueError('Aggregate samples must share the same config and checkpoint iteration')
    report = hphy_reliability_statistics(payloads)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
