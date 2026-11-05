import argparse
import copy
import os

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.models.utils.proca_prototype_estimator import \
    ProCAPrototypeEstimator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize ProCA source prototypes.')
    parser.add_argument('config')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--launcher', default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def extract_feature_and_logits(model, img, img_metas, feature_index):
    feats = model.extract_feat(img)
    feat = feats[feature_index] if isinstance(feats, (list, tuple)) else feats
    logits = model.decode_head.forward_test(feats, img_metas, model.test_cfg)
    return feat, logits


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = build_segmentor(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    source_cfg = copy.deepcopy(cfg.data.train.source)
    dataset = build_dataset(source_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)

    uda_cfg = cfg.uda
    feat_estimator = ProCAPrototypeEstimator(
        uda_cfg.num_classes, uda_cfg.feature_dim, uda_cfg.ignore_index,
        uda_cfg.use_momentum, uda_cfg.prototype_momentum).cuda()
    out_estimator = ProCAPrototypeEstimator(
        uda_cfg.num_classes, uda_cfg.num_classes, uda_cfg.ignore_index,
        uda_cfg.use_momentum, uda_cfg.prototype_momentum).cuda()

    with torch.no_grad():
        for data in data_loader:
            img = data['img'].data[0].cuda()
            img_metas = data['img_metas'].data[0]
            labels = data['gt_semantic_seg'].data[0].cuda()
            feat, logits = extract_feature_and_logits(
                model, img, img_metas, uda_cfg.feature_index)
            feat_estimator.update(feat, labels)
            out_estimator.update(logits, labels)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(feat_estimator.state_for_save(),
               os.path.join(args.output_dir, 'prototype_feat_dist.pth'))
    torch.save(out_estimator.state_for_save(),
               os.path.join(args.output_dir, 'prototype_out_dist.pth'))
    print('Saved ProCA source prototypes:')
    print(os.path.join(args.output_dir, 'prototype_feat_dist.pth'))
    print(os.path.join(args.output_dir, 'prototype_out_dist.pth'))


if __name__ == '__main__':
    main()
