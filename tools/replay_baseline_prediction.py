"""Export the Iter-4000 HRDA-3VI Baseline Student using the shared replay path."""
import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_hphy_mechanism_figure import (
    CROP, GOJI, PALETTE, colored_pixels, file_sha256, plt, save_figure, synthetic_label)

CONFIG = ROOT/'configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py'
CHECKPOINT = Path('/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi/iter_4000.pth')
OUTPUT_ROOT = Path('/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/baseline_mechanism_26rgb2_30rgb')
LOGGER = logging.getLogger('baseline_prediction')


def export_sample(arrays, metadata, output, dpi=600):
    """Save class IDs separately from palette RGB, with an exact center crop."""
    prediction, rgb = arrays['baseline_prediction'], arrays['image_rgb']
    if (prediction.shape != (256, 256) or prediction.dtype != np.uint8 or
            not np.isin(prediction, range(5)).all()):
        raise ValueError('Baseline prediction must be native uint8 class IDs 0..4')
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected original uint8 256x256 RGB')
    if dpi <= 0:
        raise ValueError('DPI must be positive')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out/'baseline_prediction_full_256.npy', prediction)
    Image.fromarray(prediction).save(out/'baseline_prediction_full_256.png')
    pixels = colored_pixels(arrays, 'baseline_prediction', metadata)
    Image.fromarray(pixels).save(out/'baseline_prediction_color_256.png', dpi=(dpi, dpi))
    Image.fromarray(pixels[30:226, 30:226]).save(out/'baseline_prediction_color_196.png', dpi=(dpi, dpi))
    Image.fromarray(rgb[30:226, 30:226]).save(out/'target_rgb_196.png', dpi=(dpi, dpi))
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(2.05, 2.4))
        axis = fig.add_axes([.1, .22, .8, .8*2.05/2.4])
        axis.imshow(pixels[30:226, 30:226], interpolation='nearest')
        axis.set_axis_off()
        axis.set_title('Baseline Student\nprediction', fontsize=9, pad=5)
        synthetic_label(fig, metadata)
        fig.savefig(out/'baseline_prediction_panel.pdf', facecolor='white')
        save_figure(fig, out/'baseline_prediction_panel.png', dpi)
    np.savez_compressed(out/'baseline_prediction.npz', **arrays,
        metadata=np.array(json.dumps(metadata, sort_keys=True)))
    metadata = dict(metadata, arrays_sha256={
        'baseline_prediction.npz': file_sha256(out/'baseline_prediction.npz')})
    (out/'baseline_meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return out


def replay(config=CONFIG, checkpoint=CHECKPOINT, out_dir=None, device='cuda:0',
           target_files=None, dpi=600):
    """Load one Baseline, then reuse the paired replay's Student-only inference."""
    import torch
    from mmcv import Config
    from mmseg.utils.mechanism import load_probe_sample, student_test_metadata
    from tools import replay_hphy_mechanism as shared

    config, checkpoint = Path(config).resolve(), Path(checkpoint).resolve()
    cfg = Config.fromfile(str(config))
    shared.verify_protocol(cfg, physiology=False)
    if cfg.model.type != 'HRDAEncoderDecoder' or cfg.model.decode_head.type != 'HRDAHead':
        raise ValueError('The independent Baseline must use the existing HRDA model and decoder')
    targets = [Path(path).resolve() for path in (shared.TARGET_FILES if target_files is None else target_files)]
    if not targets or len({path.stem for path in targets}) != len(targets):
        raise ValueError('Supply one or more targets with unique basenames')
    if dpi <= 0:
        raise ValueError('DPI must be positive')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    out = Path(out_dir) if out_dir is not None else OUTPUT_ROOT/(
        'iter_4000_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    handler = logging.FileHandler(out/'logs/replay.log', encoding='utf-8')
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    try:
        config_hash = file_sha256(config)
        LOGGER.info('Loading Baseline Student from %s', checkpoint)
        model, information = shared.load_model(cfg, checkpoint, physiology=False, device=device, work_dir=out)
        if information['completed_iter'] != 4000:
            raise ValueError('This Baseline export requires a verified Iter 4000 checkpoint')
        information = {key: value for key, value in information.items()
                       if key not in ('physiology_precision', 'hphy_probability_dtype')}
        source_files = ('tools/replay_baseline_prediction.py', 'tools/replay_hphy_mechanism.py',
            'tools/replay_doris_mechanisms.py', 'mmseg/utils/mechanism.py',
            'mmseg/models/uda/dacs.py', 'mmseg/models/uda/uda_decorator.py', 'mmseg/vi_utils.py',
            'mmseg/datasets/goji.py', 'mmseg/datasets/goji_meta.py',
            'mmseg/datasets/pipelines/loading.py', 'mmseg/datasets/pipelines/transforms.py',
            'mmseg/models/segmentors/hrda_encoder_decoder.py', 'mmseg/models/decode_heads/hrda_head.py')
        source_hashes = {name: file_sha256(ROOT/name) for name in source_files}
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
        rows = []
        for target in targets:
            probe = shared.sample_config(cfg, target)
            suffix = cfg.data.train.target.get('index_suffix', '_dir')
            vi_paths = {name: target.parents[2]/(name+suffix)/target.parent.name/target.name
                        for name in ('ndvi', 'ndre', 'cire')}
            hashes = {str(path): file_sha256(path) for path in [target, *vi_paths.values()]}
            LOGGER.info('Predicting %s, Student Iter 4000', target.name)
            sample, _ = load_probe_sample(probe, model)
            arrays, state_hash = shared.infer_model(model, sample, probe, physiology=False)
            image, vi, metas = sample
            norm = metas[0]['img_norm_cfg']
            original = image[0].numpy().transpose(1, 2, 0)*np.asarray(norm['std'])+np.asarray(norm['mean'])
            with Image.open(target) as tif:
                if tif.mode != 'RGB':
                    raise ValueError('Expected a true three-band uint8 RGB TIFF')
                rgb = np.asarray(tif).copy()
            if rgb.shape != (256, 256, 3) or not np.allclose(original[..., :3], rgb, atol=1e-3, rtol=0):
                raise ValueError('RGB pixels differ from the model input')
            valid = np.ones((256, 256), dtype=bool)
            top, bottom, left, right = probe['margins']
            if top: valid[:top] = False
            if bottom: valid[-bottom:] = False
            if left: valid[:, :left] = False
            if right: valid[:, -right:] = False
            arrays.update(image_rgb=rgb, image_6ch=image[0].numpy().transpose(1, 2, 0).copy(), valid_mask=valid)
            for index, name in enumerate(('ndvi', 'ndre', 'cire')):
                arrays[name] = original[..., index+3].astype(np.float32)
                arrays['vi_'+name] = vi[0, index].numpy().copy()
            inference = student_test_metadata(model, image, metas)
            inference['prediction_field'] = 'baseline_prediction'
            metadata = dict(direction='goji26RGB2goji30RGB', sample_id=target.name,
                target_file=str(target), target_sha256=hashes[str(target)],
                vi_files={name: dict(path=str(path), sha256=hashes[str(path)]) for name, path in vi_paths.items()},
                checkpoint_baseline=dict(information, state_sha256_before=state_hash, state_sha256_after=state_hash,
                    eval=True, inference_mode=True), checkpoint_iter=4000,
                config_baseline=dict(path=str(config), sha256=config_hash,
                    resolved_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest()),
                git_commit=commit, source_sha256=source_hashes,
                model_variant='HRDA-3VI Baseline', network_type=cfg.model.type,
                backbone=cfg.model.backbone.type, modules=dict(phy=False, struct=False, proto=False, edge=False),
                prediction_network='Student', prediction_mode='test', capture_mode='checkpoint_replay',
                student_inference=inference, input_channel_order=['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE'],
                target_normalization=dict(cfg.target_img_norm_cfg),
                vi_normalization=dict(cfg.data.train.target.vi_norm_cfg),
                image_6ch_definition='Normalized segmentation input in HWC order',
                valid_mask_definition='Historical spatial pseudo-supervision margins, independent of prediction IDs',
                class_names=list(GOJI['GOJI_CLASSES']), class_palette=PALETTE.tolist(),
                reduce_zero_label=True, ignore_index=255, original_shape=[256, 256], crop=list(CROP), center_shape=[196, 196],
                dataset_split=target.parent.name, tta=False, gt_loaded=False,
                synthetic_fixture=information['synthetic_fixture'], dpi=dpi)
            if hashes != {path: file_sha256(path) for path in hashes}:
                raise RuntimeError('An input file changed during replay')
            folder = export_sample(arrays, metadata, out/target.stem, dpi)
            rows.append(dict(sample_id=target.name, target_file=str(target), output=str(folder),
                target_sha256=hashes[str(target)], prediction_sha256=file_sha256(folder/'baseline_prediction_full_256.npy')))
        if file_sha256(config) != config_hash or file_sha256(checkpoint) != information['sha256']:
            raise RuntimeError('Configuration or checkpoint changed during replay')
        summary = dict(checkpoint=information, sample_count=len(rows), samples=rows,
            prediction_network='Student', checkpoint_iter=4000, gt_loaded=False, tta=False,
            capture_mode='checkpoint_replay', git_commit=commit)
        (out/'baseline_run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        LOGGER.info('Completed %d Baseline Student samples in %s', len(rows), out)
        return out
    except Exception:
        LOGGER.exception('Baseline Student replay failed')
        raise
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--target-file', type=Path, nargs='+', dest='target_files',
                        help='Defaults to the six TARGET_FILES in replay_hphy_mechanism.py')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    print('Saved Baseline Student predictions in '+str(replay(**vars(args))))


if __name__ == '__main__':
    main()
