"""Read-only preflight for the existing Goji experiments on a remote GPU.

No packages are installed and no datasets, configs, or checkpoints are changed.
--data-only checks uploaded files without importing PyTorch or MMCV.
"""

import argparse
import copy
from pathlib import Path
import platform
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MODALITIES = ('img_dir', 'rgb_dir', 'ann_dir', 'ndvi_dir', 'ndre_dir',
              'cire_dir')
SPLITS = ('train', 'val', 'test')


def check_uploads(root, samples):
    import numpy as np
    from PIL import Image

    weight = root / 'pretrained' / 'mit_b5.pth'
    if not weight.is_file() or weight.stat().st_size == 0:
        raise FileNotFoundError('Missing or empty MiT-B5 weight: {}'.format(weight))
    print('Weight: {} ({} bytes)'.format(weight, weight.stat().st_size))
    labels_seen = set()
    for area in ('26', '30'):
        for split in SPLITS:
            folders = {
                name: root / 'mydataset' / area / name / split
                for name in MODALITIES
            }
            files = {}
            for name, folder in folders.items():
                files[name] = {
                    str(path.relative_to(folder)): path
                    for path in folder.rglob('*.tif')
                }
                if not files[name]:
                    raise FileNotFoundError('No lowercase .tif files: {}'.format(folder))
            names = set(files['img_dir'])
            for name, mapping in files.items():
                missing = sorted(names - set(mapping))
                extra = sorted(set(mapping) - names)
                if missing or extra:
                    raise ValueError('{}: filename mismatch; missing={}, extra={}'.format(
                        folders[name], missing[:5], extra[:5]))

            ordered = sorted(names)
            count = min(samples, len(ordered))
            positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
            for position in positions:
                filename = ordered[position]
                arrays = {}
                for name in MODALITIES:
                    with Image.open(files[name][filename]) as image:
                        arrays[name] = np.array(image)
                shape = arrays['img_dir'].shape[:2]
                for name, array in arrays.items():
                    if array.shape[:2] != shape or shape != (256, 256):
                        raise ValueError('{}: expected aligned 256x256 patches, got {}'.format(
                            files[name][filename], array.shape))
                    if not np.isfinite(array).all():
                        raise ValueError('NaN/Inf: {}'.format(files[name][filename]))
                    if name in ('img_dir', 'rgb_dir'):
                        if array.ndim != 3 or array.shape[2] != 3:
                            raise ValueError('Expected three image channels: {}'.format(
                                files[name][filename]))
                    elif array.ndim != 2:
                        raise ValueError('Expected a single-band TIFF: {}'.format(
                            files[name][filename]))
                values = set(np.unique(arrays['ann_dir']).tolist())
                if not values <= {0, 1, 2, 3, 4, 5, 255}:
                    raise ValueError('Unexpected raw label values {}: {}'.format(
                        sorted(values), files['ann_dir'][filename]))
                labels_seen.update(values)
            print('Area {} / {}: {} matched filenames, {} content samples'.format(
                area, split, len(names), count))
    print('Sampled raw label values: {}'.format(sorted(labels_seen)))
    if 5 not in labels_seen:
        print('WARNING: no raw class 5 in samples. Confirm masks are original '
              '1..5, not already reduced 0..4. Absence in samples is inconclusive.')
    print('File matching passed; TIFF contents were sampled, not fully audited.')


def check_environment():
    import torch
    import torch.nn.functional as functional

    print('Python: {} ({})'.format(sys.version, sys.executable))
    print('OS: {}'.format(platform.platform()))
    print('Torch: {}; CUDA runtime: {}'.format(torch.__version__, torch.version.cuda))
    print('cuDNN: {}; CUDA available: {}'.format(
        torch.backends.cudnn.version(), torch.cuda.is_available()))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check driver/container access first.')
    print('GPU: {}; capability: {}; wheel arch list: {}'.format(
        torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0),
        torch.cuda.get_arch_list()))

    import mmcv
    import timm
    import torchvision
    import kornia
    import mmseg.models  # noqa: F401 - exercise registrations/import dependencies
    from mmcv.parallel import DataContainer, MMDataParallel, collate

    print('MMCV: {}; timm: {}; torchvision: {}; Kornia: {}'.format(
        mmcv.__version__, timm.__version__, torchvision.__version__,
        kornia.__version__))
    print('Local MMSeg import: {}'.format(mmseg.models.__file__))
    # Exercise the same DataContainer CPU->CUDA scatter used in training.
    network = MMDataParallel(
        torch.nn.Conv2d(6, 5, 3, padding=1).cuda(), device_ids=[0])
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    batch = collate([
        dict(img=DataContainer(torch.randn(6, 32, 32), stack=True))
        for _ in range(2)
    ], samples_per_gpu=2)
    logits = network(batch['img'])
    logits = functional.interpolate(
        logits, size=(64, 64), mode='bilinear', align_corners=False)
    labels = torch.randint(0, 5, (2, 64, 64), device='cuda')
    loss = functional.cross_entropy(logits, labels)
    loss.backward()
    if not torch.isfinite(loss).item() or not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in network.parameters()):
        raise RuntimeError('Nonfinite CUDA loss/gradients.')
    optimizer.step()
    left = torch.randn(128, 128, device='cuda', requires_grad=True)
    right = torch.randn(128, 128, device='cuda')
    matmul_loss = (left @ right).square().mean()
    matmul_loss.backward()
    if not torch.isfinite(matmul_loss).item() or not torch.isfinite(left.grad).all().item():
        raise RuntimeError('Nonfinite CUDA matmul result/gradients.')
    torch.cuda.synchronize()
    print('CUDA conv / scatter / interpolation / CE / matmul / backward / AdamW passed.')


def check_pipeline(config, root):
    from mmcv import Config
    from mmseg.datasets import build_dataset

    cfg = Config.fromfile(str(config))
    print('Config: {}'.format(config))
    print('Configured pretrained path: {}'.format(cfg.model.backbone.pretrained))
    print('Training commands must override it to: {}'.format(
        root / 'pretrained' / 'mit_b5.pth'))
    for name in ('source', 'target'):
        print('{} root: {}'.format(name, cfg.data.train[name].data_root))
    # These are the effective config paths. Do not silently replace dataset paths.
    dataset = build_dataset(copy.deepcopy(cfg.data.train))
    batch = dataset[0]
    for name in ('img', 'target_img'):
        tensor = batch[name].data
        if tuple(tensor.shape) != (6, 256, 256):
            raise ValueError('{} has unexpected pipeline shape {}'.format(
                name, tuple(tensor.shape)))
        print('{}: {}'.format(name, tuple(tensor.shape)))
    for name in ('gt_semantic_seg', 'target_gt_semantic_seg'):
        if name in batch:
            print('{} reduced labels: {}'.format(
                name, batch[name].data.unique().tolist()))
    validation = build_dataset(copy.deepcopy(cfg.data.val), dict(test_mode=True))
    validation[0]
    print('Train/validation pipeline samples passed; no HRDA model training was run.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/shared-nvme'))
    parser.add_argument('--config', type=Path, default=Path(
        'configs/hrda/goji26RGB2goji30RGB_hrda_3vi.py'))
    parser.add_argument('--samples-per-split', type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--env-only', action='store_true')
    mode.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error('--samples-per-split must be positive')
    if not args.data_only:
        check_environment()
    if not args.env_only:
        check_uploads(args.root, args.samples_per_split)
        if not args.data_only:
            config = args.config if args.config.is_absolute() else REPO / args.config
            check_pipeline(config, args.root)
    print('Requested preflight stages passed. Full remote training remains unverified.')


if __name__ == '__main__':
    main()
