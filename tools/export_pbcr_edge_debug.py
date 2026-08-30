import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from PIL import Image

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from tools.test import update_legacy_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export PBCR edge response maps for figure analysis.')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='directory where PBCR response maps will be saved')
    parser.add_argument(
        '--split',
        choices=['test', 'val'],
        default='test',
        help='dataset split to export')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50,
        help='maximum number of samples to export')
    parser.add_argument(
        '--inference-mode',
        choices=['same', 'whole', 'slide'],
        default='same',
        help='inference mode override, matching tools/test.py')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    return parser.parse_args()


def save_heatmap(data, out_file, cmap='jet'):
    data = data.detach().float().cpu().numpy()
    if data.ndim == 3:
        data = data.squeeze()
    data = np.nan_to_num(data)
    data = np.clip(data, 0.0, 1.0)
    color = (cm.get_cmap(cmap)(data)[..., :3] * 255).astype(np.uint8)
    mmcv.mkdir_or_exist(osp.dirname(out_file))
    Image.fromarray(color, mode='RGB').save(out_file)


def save_color_seg(seg, palette, out_file):
    palette = np.asarray(palette, dtype=np.uint8)
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label] = color
    mmcv.mkdir_or_exist(osp.dirname(out_file))
    Image.fromarray(color_seg, mode='RGB').save(out_file)


def save_input(img_tensor, img_meta, out_file):
    img_norm_cfg = img_meta['img_norm_cfg']
    vis_tensor = img_tensor[:3].detach().cpu()
    img = tensor2imgs(
        vis_tensor.unsqueeze(0),
        mean=img_norm_cfg['mean'][:3],
        std=img_norm_cfg['std'][:3],
        to_rgb=img_norm_cfg.get('to_rgb', False))[0]
    vis_channel_order = img_meta['img_norm_cfg'].get('vis_channel_order', None)
    if vis_channel_order is not None:
        img = img[..., vis_channel_order]
    h, w, _ = img_meta['img_shape']
    img = img[:h, :w, :]
    ori_h, ori_w = img_meta['ori_shape'][:-1]
    img = mmcv.imresize(img, (ori_w, ori_h))
    mmcv.mkdir_or_exist(osp.dirname(out_file))
    mmcv.imwrite(img, out_file)


def unwrap_data_container(value):
    if hasattr(value, 'data'):
        return value.data
    return value


def get_batch_img(data, device):
    img = data['img']
    if isinstance(img, (list, tuple)):
        img = img[0]
    return img.to(device)


def get_batch_img_metas(data):
    img_metas = data['img_metas']
    if isinstance(img_metas, (list, tuple)):
        img_metas = img_metas[0]
    img_metas = unwrap_data_container(img_metas)
    if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and \
            isinstance(img_metas[0], (list, tuple)):
        img_metas = img_metas[0]
    return img_metas


def build_img_edge_response(img):
    eps = 1e-6
    x = img.float()
    channels = x.shape[1]
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    x = (x - x_min) / (x_max - x_min + eps)
    sobel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gx = F.conv2d(x, sobel_x, padding=1, groups=channels)
    gy = F.conv2d(x, sobel_y, padding=1, groups=channels)
    edge = torch.sqrt(gx.pow(2) + gy.pow(2) + eps).mean(dim=1, keepdim=True)
    norm = edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (edge / norm).clamp(0.0, 1.0)


def build_prob_edge_response(probs):
    eps = 1e-6
    channels = probs.shape[1]
    sobel_x = probs.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = probs.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gx = F.conv2d(probs.float(), sobel_x, padding=1, groups=channels)
    gy = F.conv2d(probs.float(), sobel_y, padding=1, groups=channels)
    edge = torch.sqrt(gx.pow(2) + gy.pow(2) + eps).mean(dim=1, keepdim=True)
    norm = edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (edge / norm).clamp(0.0, 1.0)


def find_checkpoint_tensor(checkpoint, suffix):
    state_dict = checkpoint.get('state_dict', checkpoint)
    for key, value in state_dict.items():
        if key.endswith(suffix):
            return value
    return None


def build_boundary_pair_response(label, checkpoint, num_classes, min_weight,
                                 power):
    pair_prior = find_checkpoint_tensor(checkpoint, 'struct_edge_pair_prior')
    ready = find_checkpoint_tensor(checkpoint, 'struct_edge_pair_prior_ready')
    if pair_prior is None or (ready is not None and not bool(ready.item())):
        return label.new_zeros((label.shape[0], 1, label.shape[1],
                                label.shape[2])).float()

    pair_prior = pair_prior.to(label.device).float()
    pair_prior = pair_prior / pair_prior.sum(dim=1, keepdim=True).clamp_min(
        1e-6)
    valid = (label >= 0) & (label < num_classes)
    response = label.new_zeros(label.shape).float()
    count = label.new_zeros(label.shape).float()

    for dy, dx in ((0, 1), (1, 0)):
        shifted = torch.roll(label, shifts=(-dy, -dx), dims=(-2, -1))
        boundary = valid & torch.roll(valid, shifts=(-dy, -dx), dims=(-2, -1))
        boundary = boundary & (label != shifted)
        score = pair_prior[label.clamp(0, num_classes - 1),
                           shifted.clamp(0, num_classes - 1)]
        score = score.clamp(0.0, 1.0).pow(power)
        response = response + torch.where(boundary, score, score.new_zeros(()))
        count = count + boundary.float()

    response = response / count.clamp_min(1.0)
    gate = min_weight + (1.0 - min_weight) * response
    response = (gate - min_weight) / max(1.0 - min_weight, 1e-6)
    return response.unsqueeze(1).clamp(0.0, 1.0)


def prepare_cfg(cfg, inference_mode):
    cfg = update_legacy_cfg(cfg)
    cfg.model.pretrained = None
    if inference_mode == 'whole':
        cfg.model.test_cfg.mode = 'whole'
    elif inference_mode == 'slide':
        cfg.model.test_cfg.mode = 'slide'
        crsize = cfg.data.train.get('sync_crop_size', cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    return cfg


def main():
    args = parse_args()
    cfg = prepare_cfg(mmcv.Config.fromfile(args.config), args.inference_mode)

    dataset_cfg = cfg.data[args.split]
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES',
                                                   dataset.CLASSES)
    model.PALETTE = checkpoint.get('meta', {}).get('PALETTE',
                                                   dataset.PALETTE)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    edge_cfg = cfg.uda.get('struct_edge_cfg', {})
    prob_w = float(edge_cfg.get('prob_edge_weight', 0.4))
    img_w = float(edge_cfg.get('img_edge_weight', 0.4))
    pair_w = float(edge_cfg.get('pair_edge_weight', 0.2))
    min_weight = float(edge_cfg.get('boundary_pair_min_weight', 0.60))
    power = float(edge_cfg.get('boundary_pair_power', 0.5))
    num_classes = int(edge_cfg.get('num_classes', len(dataset.CLASSES)))

    mmcv.mkdir_or_exist(args.out_dir)
    prog_bar = mmcv.ProgressBar(min(len(dataset), args.max_samples))
    for idx in range(min(len(dataset), args.max_samples)):
        data = collate([dataset[idx]], samples_per_gpu=1)
        if device.type == 'cuda':
            data = scatter(data, [device.index or 0])[0]
        img = get_batch_img(data, device)
        img_metas = get_batch_img_metas(data)
        img_meta = img_metas[0]

        with torch.no_grad():
            probs = model.inference(img, img_metas, rescale=False)
            pred = probs.argmax(dim=1)
            confidence = probs.max(dim=1, keepdim=True)[0]
            img_edge = build_img_edge_response(img)
            prob_edge = build_prob_edge_response(probs)
            if prob_edge.shape[-2:] != img_edge.shape[-2:]:
                prob_edge = F.interpolate(
                    prob_edge,
                    size=img_edge.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            pair_response = build_boundary_pair_response(
                pred, checkpoint, num_classes, min_weight, power)
            if pair_response.shape[-2:] != img_edge.shape[-2:]:
                pair_response = F.interpolate(
                    pair_response, size=img_edge.shape[-2:], mode='nearest')
            struct_edge_gate = (
                prob_w * prob_edge + img_w * img_edge +
                pair_w * pair_response).clamp(0.0, 1.0)

        stem = osp.splitext(osp.basename(img_meta['ori_filename']))[0]
        prefix = osp.join(args.out_dir, f'{idx:04d}_{stem}')
        save_input(img[0], img_meta, prefix + '_input.png')
        save_heatmap(prob_edge[0, 0], prefix + '_prob_edge.png')
        save_heatmap(img_edge[0, 0], prefix + '_img_edge.png')
        save_heatmap(pair_response[0, 0],
                     prefix + '_boundary_pair_response.png')
        save_heatmap(struct_edge_gate[0, 0],
                     prefix + '_struct_edge_gate.png')
        save_heatmap(confidence[0, 0], prefix + '_confidence.png')
        save_color_seg(pred[0].detach().cpu().numpy(), dataset.PALETTE,
                       prefix + '_prediction.png')
        prog_bar.update()


if __name__ == '__main__':
    main()
