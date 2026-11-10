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
        response_supervision='Goji and other vegetation share the positive label of the physiology head',
        annotation_usage='Offline reference and class summaries after model forward',
        statistics_scope='Valid pixels grouped by existing validation ground truth',
        ignore_index=255, classes=rows)
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
