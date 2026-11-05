import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from PIL import Image

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate ProCA class-wise target pseudo labels.')
    parser.add_argument('config')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bins', type=int, default=1000)
    parser.add_argument('--launcher', default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def build_target_train_inference_dataset(cfg):
    target_cfg = copy.deepcopy(cfg.data.train.target)
    if 'val' in cfg.data and 'pipeline' in cfg.data.val:
        target_cfg.pipeline = copy.deepcopy(cfg.data.val.pipeline)
    target_cfg.test_mode = True
    return build_dataset(target_cfg)


def rel_output_path(img_meta):
    rel = img_meta.get('ori_filename', img_meta.get('filename'))
    rel = rel.replace('\\', '/')
    if '/img_dir/' in rel:
        rel = rel.split('/img_dir/', 1)[1]
        rel = rel.split('/', 1)[1] if '/' in rel else rel
    return rel.lstrip('/')


def update_histograms(histograms, probs, labels, bins):
    probs = probs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for cls in range(histograms.shape[0]):
        values = probs[labels == cls]
        if values.size == 0:
            continue
        hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
        histograms[cls] += hist.astype(np.int64)


def thresholds_from_histograms(histograms, percent, upper_bound):
    thresholds = []
    bins = histograms.shape[1]
    for hist in histograms:
        total = int(hist.sum())
        if total == 0:
            thresholds.append(upper_bound)
            continue
        keep_count = max(int(total * percent), 1)
        cumulative = np.cumsum(hist[::-1])
        rev_idx = int(np.searchsorted(cumulative, keep_count, side='left'))
        bin_idx = bins - 1 - rev_idx
        threshold = float(bin_idx) / float(bins)
        thresholds.append(min(threshold, upper_bound))
    return np.asarray(thresholds, dtype=np.float32)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = build_segmentor(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    dataset = build_target_train_inference_dataset(cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)

    num_classes = cfg.uda.num_classes
    histograms = np.zeros((num_classes, args.bins), dtype=np.int64)
    with torch.no_grad():
        for data in data_loader:
            img = data['img'][0].data[0].cuda()
            img_metas = data['img_metas'][0].data[0]
            logits = model.encode_decode(img, img_metas)
            prob = torch.softmax(logits, dim=1)
            conf, label = prob.max(dim=1)
            update_histograms(histograms, conf, label, args.bins)

    pipe_cfg = cfg.proca_pipeline
    thresholds = thresholds_from_histograms(
        histograms, pipe_cfg.threshold_percent,
        pipe_cfg.threshold_upper_bound)

    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for data in data_loader:
            img = data['img'][0].data[0].cuda()
            img_metas = data['img_metas'][0].data[0]
            logits = model.encode_decode(img, img_metas)
            prob = torch.softmax(logits, dim=1)
            conf, label = prob.max(dim=1)
            pseudo = label.squeeze(0).detach().cpu().numpy().astype(np.uint8)
            conf_np = conf.squeeze(0).detach().cpu().numpy()
            for cls in range(num_classes):
                mask = (pseudo == cls) & (conf_np < thresholds[cls])
                pseudo[mask] = pipe_cfg.pseudo_label_ignore_index
            rel = rel_output_path(img_metas[0])
            out_path = os.path.join(args.output_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(pseudo).save(out_path)

    np.save(os.path.join(args.output_dir, 'class_thresholds.npy'), thresholds)
    print('Saved ProCA pseudo labels to:', args.output_dir)
    print('Class thresholds:', thresholds.tolist())


if __name__ == '__main__':
    main()
