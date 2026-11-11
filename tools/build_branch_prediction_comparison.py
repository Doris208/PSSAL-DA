"""Compare saved Baseline, H-Phy and Structural Students without model inference."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_hphy_mechanism_figure import (
    CROP, GOJI, PALETTE, colored_pixels, file_sha256, plt, save_figure, synthetic_label)

ROLES = ('baseline', 'phy', 'struct')
FIELDS = ('baseline_prediction', 'hphy_prediction', 'struct_prediction')
TITLES = ('Baseline Student', 'H-Phy Student', 'Structural Student')


def load_artifact(path, role):
    path = Path(path).resolve()
    names = dict(baseline=('baseline_meta.json',),
        phy=('phy_mechanism_meta.json', 'mechanism_meta.json'),
        struct=('struct_mechanism_meta.json', 'mechanism_meta.json'))
    metadata_path = next((path.parent/name for name in names[role] if (path.parent/name).is_file()), None)
    if metadata_path is None:
        raise ValueError('Missing original replay metadata beside '+str(path))
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    if meta.get('arrays_sha256', {}).get(path.name) != file_sha256(path):
        raise ValueError('NPZ hash differs from original replay provenance: '+str(path))
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files if key != 'metadata'}
    field = FIELDS[ROLES.index(role)]
    prediction = arrays.get(field)
    if (prediction is None or prediction.shape != (256, 256) or prediction.dtype != np.uint8 or
            not np.isin(prediction, range(5)).all()):
        raise ValueError('Expected native uint8 Student IDs 0..4 in '+field)
    checkpoint = meta.get('checkpoint_'+role, {})
    if (meta.get('prediction_network') != 'Student' or not checkpoint.get('path') or
            not checkpoint.get('sha256') or type(checkpoint.get('completed_iter')) is not int or
            checkpoint['completed_iter'] <= 0 or not checkpoint.get('eval') or
            not checkpoint.get('inference_mode') or not checkpoint.get('state_sha256_before') or
            checkpoint.get('state_sha256_before') != checkpoint.get('state_sha256_after')):
        raise ValueError('Missing verified Student/checkpoint/read-only provenance for '+role)
    if (meta.get('original_shape') != [256, 256] or meta.get('crop') != list(CROP) or
            meta.get('center_shape') != [196, 196] or meta.get('class_palette') != PALETTE.tolist() or
            meta.get('class_names') != list(GOJI['GOJI_CLASSES']) or
            not meta.get('reduce_zero_label') or meta.get('ignore_index') != 255 or
            meta.get('input_channel_order') != ['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE']):
        raise ValueError('Spatial grid, crop, channels or class mapping differs for '+role)
    if (not meta.get('target_file') or not meta.get('target_sha256') or
            meta.get('sample_id') != Path(meta['target_file']).name or
            set(meta.get('vi_files', {})) != {'ndvi', 'ndre', 'cire'} or
            any(not record.get('sha256') for record in meta['vi_files'].values())):
        raise ValueError('Missing sample/VI file provenance for '+role)
    if meta.get('tta', False) or meta.get('student_inference', {}).get('flip', False):
        raise ValueError('Extra test-time augmentation is outside this comparison')
    if role == 'baseline' and not meta.get('student_inference'):
        raise ValueError('Use the independent Baseline replay with its recorded Student inference settings')
    for key, shape in (('image_rgb', (256, 256, 3)), ('image_6ch', (256, 256, 6))):
        value = arrays.get(key)
        if value is None or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Missing full native input array '+key)
    if arrays['image_rgb'].dtype != np.uint8:
        raise ValueError('Expected original uint8 RGB')
    return arrays, meta, dict(npz=str(path), npz_sha256=file_sha256(path),
        metadata=str(metadata_path), metadata_sha256=file_sha256(metadata_path))


def verify_alignment(artifacts):
    reference, reference_meta, _ = artifacts[0]
    inference_keys = ('test_cfg', 'scales', 'hr_crop_size', 'hr_slide_inference', 'rescale', 'fp16_enabled')
    for arrays, meta, _ in artifacts:
        for key in ('sample_id', 'target_sha256', 'direction', 'target_normalization', 'vi_normalization'):
            if meta.get(key) is None or meta[key] != reference_meta.get(key):
                raise ValueError('Mismatched sample or input protocol: '+key)
        for name in ('ndvi', 'ndre', 'cire'):
            if meta['vi_files'][name]['sha256'] != reference_meta['vi_files'][name]['sha256']:
                raise ValueError('VI input hash differs: '+name)
        for key in ('image_rgb', 'image_6ch', 'valid_mask'):
            if key not in arrays or key not in reference or not np.array_equal(arrays[key], reference[key]):
                raise ValueError('Input pixels or spatial validity differ: '+key)
        recorded = meta.get('student_inference')
        if recorded:
            if recorded.get('model_identity') != 'Student' or recorded.get('rescale') is not True:
                raise ValueError('Inference identity or rescale protocol differs')
            for key in inference_keys:
                if recorded.get(key) != reference_meta['student_inference'].get(key):
                    raise ValueError('Student inference setting differs: '+key)


def load_raw_gt(path, sample_id):
    """Apply the existing Goji raw 0/1..5/255 to reduced 255/0..4/255 mapping."""
    path = Path(path).resolve()
    if path.stem != Path(sample_id).stem:
        raise ValueError('GT basename must match the target tile')
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
    if raw.shape != (256, 256) or raw.dtype.kind not in 'iu' or not np.isin(raw, [0, 1, 2, 3, 4, 5, 255]).all():
        raise ValueError('GT must use original Goji IDs 0/1..5/255 on the native grid')
    truth = np.full(raw.shape, 255, dtype=np.uint8)
    valid = (raw >= 1) & (raw <= 5)
    truth[valid] = raw[valid]-1
    return truth


def gt_statistics(truth, predictions):
    """Report full/cropped per-tile metrics with absent unions excluded from means."""
    result = {}
    for scope, region in (('full256', np.s_[:, :]), ('center196', np.s_[30:226, 30:226])):
        gt = truth[region]
        valid = gt != 255
        values, corrections = {}, {}
        for role, prediction in zip(ROLES, predictions):
            pred = prediction[region]
            per_class = []
            for cls in range(5):
                intersection = int((valid & (gt == cls) & (pred == cls)).sum())
                union = int((valid & ((gt == cls) | (pred == cls))).sum())
                per_class.append(intersection/union if union else None)
            def mean_available(ids):
                present = [per_class[cls] for cls in ids if per_class[cls] is not None]
                return float(np.mean(present)) if present else None
            values[role] = dict(iou=dict(zip(GOJI['GOJI_CLASSES'], per_class)),
                mIoU=mean_available(range(5)), Veg_mIoU=mean_available((0, 1)),
                NonVeg_mIoU=mean_available((2, 3, 4)),
                goji_as_vegetation=int((valid & (gt == 0) & (pred == 1)).sum()),
                vegetation_as_goji=int((valid & (gt == 1) & (pred == 0)).sum()))
        baseline_correct = predictions[0][region] == gt
        for role, pred in zip(ROLES[1:], predictions[1:]):
            correct = pred[region] == gt
            corrections[role] = {name: dict(
                corrected=int((mask & ~baseline_correct & correct).sum()),
                regressed=int((mask & baseline_correct & ~correct).sum()),
                both_correct=int((mask & baseline_correct & correct).sum()),
                both_wrong=int((mask & ~baseline_correct & ~correct).sum()))
                for name, mask in [('all', valid)]+[(name, valid & (gt == cls))
                    for cls, name in enumerate(GOJI['GOJI_CLASSES'])]}
        result[scope] = dict(valid_gt_pixels=int(valid.sum()), models=values, changes_from_baseline=corrections)
    return dict(scopes=result, values_unit='fraction', absent_class_policy='null if union=0, excluded from mean',
        valid_region='GT != 255, independent of training supervision margins',
        interpretation='Per-tile descriptive metrics at the recorded checkpoints', dataset_evaluation=False)


def comparison_figure(panels, synthetic=False):
    width, left, gap = 7.2, .12, .075
    side = (width-2*left-(len(panels)-1)*gap)/len(panels)
    height = side+.85
    fig = plt.figure(figsize=(width, height))
    for index, (title, pixels) in enumerate(panels):
        axis = fig.add_axes([(left+index*(side+gap))/width, .18/height, side/width, side/height])
        axis.imshow(pixels, interpolation='nearest')
        axis.set_axis_off()
        axis.set_title('({}) {}'.format(chr(97+index), title), fontsize=9, pad=5)
    synthetic_label(fig, dict(synthetic_fixture=synthetic))
    return fig


def render(baseline_npz, phy_npz, struct_npz, out_dir, gt_file=None, dpi=600):
    if dpi <= 0:
        raise ValueError('DPI must be positive')
    artifacts = [load_artifact(path, role) for role, path in zip(ROLES, (baseline_npz, phy_npz, struct_npz))]
    verify_alignment(artifacts)
    predictions = [arrays[field] for (arrays, _, _), field in zip(artifacts, FIELDS)]
    truth = load_raw_gt(gt_file, artifacts[0][1]['sample_id']) if gt_file is not None else None
    panels = [('Target RGB', artifacts[0][0]['image_rgb'][30:226, 30:226])]
    if truth is not None:
        panels.append(('Ground Truth', colored_pixels({'gt_prediction': truth}, 'gt_prediction', {})[30:226, 30:226]))
    for role, prediction, title, (_, meta, _) in zip(ROLES, predictions, TITLES, artifacts):
        iteration = meta['checkpoint_'+role]['completed_iter']
        pixels = colored_pixels({'prediction': prediction}, 'prediction', {})[30:226, 30:226]
        panels.append((title.replace(' Student', '\nStudent')+'\nIter '+str(iteration), pixels))
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    synthetic = any(meta.get('synthetic_fixture', False) for _, meta, _ in artifacts)
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42}):
        fig = comparison_figure(panels, synthetic)
        fig.savefig(out/'branch_prediction_comparison.pdf', facecolor='white')
        save_figure(fig, out/'branch_prediction_comparison.png', dpi)
    for role, prediction in zip(ROLES, predictions):
        pixels = colored_pixels({'prediction': prediction}, 'prediction', {})[30:226, 30:226]
        Image.fromarray(pixels).save(out/(role+'_student_color_196.png'), dpi=(dpi, dpi))
    metadata = dict(sample_id=artifacts[0][1]['sample_id'], crop=list(CROP),
        source_data={role: dict(source, checkpoint=meta['checkpoint_'+role],
            student_inference=meta.get('student_inference'),
            inference_details_recorded='student_inference' in meta,
            inference_settings_check=('matched against Baseline' if meta.get('student_inference')
                else 'Historical replay does not record detailed Student inference settings'))
            for role, (_, meta, source) in zip(ROLES, artifacts)},
        class_names=list(GOJI['GOJI_CLASSES']), class_palette=PALETTE.tolist(),
        prediction_network='Student', gt_loaded=truth is not None,
        gt=None if truth is None else dict(path=str(Path(gt_file).resolve()), sha256=file_sha256(gt_file),
            mapping='Raw Goji labels, reduce_zero_label=True'),
        synthetic_fixture=synthetic,
        interpretation='Specified-checkpoint qualitative comparison with visible training iterations')
    if truth is not None:
        statistics = gt_statistics(truth, predictions)
        (out/'tile_metrics.json').write_text(json.dumps(statistics, indent=2), encoding='utf-8')
    for _, _, source in artifacts:
        if (file_sha256(source['npz']) != source['npz_sha256'] or
                file_sha256(source['metadata']) != source['metadata_sha256']):
            raise RuntimeError('Saved prediction provenance changed during comparison')
    (out/'comparison_meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument('--'+role+'-npz', type=Path, required=True)
    parser.add_argument('--gt-file', type=Path, help='Original Goji labels, 0/1..5/255, matching the target tile basename')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=600)
    args = vars(parser.parse_args(argv))
    print('Saved Student comparison in '+str(render(**args)))


if __name__ == '__main__':
    main()
