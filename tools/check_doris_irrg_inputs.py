"""Verify real IRRG26/RGB30 files and the production six-channel input pipeline."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_gate.py'


def check_sample(dataset_config, normalization, filename, channel_names):
    from mmseg.datasets import build_dataset
    from mmseg.utils.mechanism import exact_probe_index
    from mmseg.vi_utils import prepare_goji_index

    dataset_config = deepcopy(dict(dataset_config))
    dataset_config.update(test_mode=True, ann_dir=None, split=None, pipeline=[
        dict(type='LoadImageFromFile'), dict(type='LoadGojiIndices'),
        dict(type='Normalize', **normalization), dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'vi_img'])])
    dataset_config.pop('crop_pseudo_margins', None)
    root = Path(dataset_config['data_root'])
    image = root / dataset_config['img_dir'] / filename
    split = image.parent.name
    vi_paths = {name: root / (name + dataset_config.get('index_suffix', '_dir')) / split / filename
                for name in dataset_config['vi_names']}
    paths = [image, *vi_paths.values()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing input files: ' + ', '.join(missing))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    dataset = build_dataset(dataset_config)
    item = dataset[exact_probe_index(dataset, filename)]
    tensor = item['img'].data.numpy().transpose(1, 2, 0)
    vi = item['vi_img'].data.numpy()
    with Image.open(image) as raw:
        pixels = np.asarray(raw).copy()
    expected = [pixels]
    for path in vi_paths.values():
        with Image.open(path) as raw:
            expected.append(prepare_goji_index(np.asarray(raw))[..., None])
    expected = np.concatenate(expected, axis=-1)
    actual = tensor * np.asarray(normalization['std']) + np.asarray(normalization['mean'])
    if (tensor.shape != (256, 256, 6) or vi.shape != (3, 256, 256) or
            not np.isfinite(tensor).all() or not np.isfinite(vi).all() or
            not np.allclose(actual, expected, atol=1e-3, rtol=0)):
        raise ValueError('Production loader/VI/Normalize tensor differs from the native six bands')
    if hashes != {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in hashes}:
        raise RuntimeError('Input file changed during validation')
    return dict(filename=filename, image=str(image), vi_files={k: str(v) for k, v in vi_paths.items()},
        file_sha256=hashes, tensor_shape=list(tensor.shape), vi_shape=list(vi.shape),
        normalization=dict(normalization), channel_order=channel_names,
        tensor_sha256=hashlib.sha256(tensor.tobytes()).hexdigest(), passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/root/shared-nvme/mydataset'))
    parser.add_argument('--source-filename', help='Defaults to the first source training TIFF')
    parser.add_argument('--output', type=Path, help='Optional new JSON report')
    args = parser.parse_args()
    from mmcv import Config
    config = Config.fromfile(str(CONFIG))
    source = deepcopy(config.data.train.source)
    source.data_root = str(args.data_root / '26')
    source_files = sorted((Path(source.data_root) / source.img_dir).glob('*.tif'))
    if args.source_filename is None and not source_files:
        raise FileNotFoundError('No source IRRG26 training TIFF in ' + source.data_root)
    filename = args.source_filename or source_files[0].name
    rows = [check_sample(source, config.source_img_norm_cfg, filename,
                         ['NIR', 'R', 'G', 'NDVI', 'NDRE', 'CIRE'])]
    target = deepcopy(config.data.train.target)
    target.update(data_root=str(args.data_root / '30'), img_dir='rgb_dir/val')
    for filename in config.uda.fixed_probe_filenames:
        rows.append(check_sample(target, config.target_img_norm_cfg, filename,
                                 ['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE']))
    report = dict(direction='goji26IRRG2goji30RGB', samples=rows, gt_loaded=False,
        source_semantics='Stored band identities NIR/R/G follow the existing dataset contract',
        verification='Production tensors compared to native stored bands and VI files')
    if args.output is not None:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
