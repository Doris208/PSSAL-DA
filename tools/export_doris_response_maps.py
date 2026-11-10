"""Export two aligned DORIS response strips from fixed target samples.

Uses the full UDA checkpoint (including EMA Teacher and source-only priors).
No annotation is read to update priors; no training forward is executed.
"""
import argparse
import copy
import json
import os.path as osp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F


def load_response_checkpoint(model, checkpoint, allow_missing_priors=False):
    """Restore full UDA state, including lazily allocated prototype buffers."""
    state = checkpoint.get('state_dict', checkpoint)
    state = {(key[7:] if key.startswith('module.') else key): value
             for key, value in state.items()}
    required = ('struct_gate.source_adj_ema', 'struct_gate.source_adj_initialized',
                'struct_gate.semantic_proto_ema', 'struct_gate.semantic_proto_initialized',
                'struct_edge_pair_prior', 'struct_edge_pair_prior_ready')
    missing = [key for key in required if key not in state]
    if missing and not allow_missing_priors:
        raise ValueError('Checkpoint is missing required source priors: ' + ', '.join(missing))
    teacher_keys = [key for key in model.state_dict() if key.startswith('ema_model.')]
    missing_teacher = [key for key in teacher_keys if key not in state]
    if not teacher_keys or missing_teacher:
        raise ValueError('A full UDA checkpoint with EMA Teacher weights is required; '
                         'missing Teacher keys: ' + ', '.join(missing_teacher[:5]))
    # ProtoGate allocates these buffers on its first real feature batch.
    for name, module in model.named_modules():
        if name != 'proto_gate':
            continue
        for buffer_name, value in module.named_buffers(recurse=False):
            key = name + '.' + buffer_name
            if key in state and value.shape != state[key].shape and value.numel() == 0:
                setattr(module, buffer_name, torch.empty_like(state[key]))
    model.load_state_dict(state, strict=False)
    readiness = dict(
        source_adj_classes=int(model.struct_gate.source_adj_initialized.sum().item()),
        probability_semantic_classes=int(model.struct_gate.semantic_proto_initialized.sum().item()),
        boundary_pair_ready=bool(model.struct_edge_pair_prior_ready.item()),
        missing_prior_keys=missing)
    unready = (readiness['source_adj_classes'] == 0 or
               readiness['probability_semantic_classes'] == 0 or
               not readiness['boundary_pair_ready'])
    if unready and not allow_missing_priors:
        raise ValueError('Checkpoint source priors are uninitialized: ' + str(readiness) +
                         '; use a later checkpoint, or explicitly --allow-missing-priors.')
    print('Checkpoint prior readiness:', json.dumps(readiness, ensure_ascii=False))
    return readiness


@torch.no_grad()
def compute_responses(model, image, image_metas, spatial_valid=None):
    """Use the same Teacher and fusion methods as DORIS training, read-only."""
    logits = model.get_ema_model().generate_pseudo_label(image, image_metas)
    probabilities = torch.softmax(logits, dim=1)
    confidence, label = probabilities.max(dim=1)
    _, structural = model.struct_gate.apply(
        target_probs=probabilities, pseudo_label=label,
        pseudo_weight=torch.ones_like(confidence), target_confidence=confidence,
        source_gt=None, source_probs=None, iter=model.local_iter,
        collect_debug=False, log_stats=False, update_priors=False)
    boundary = model.doris_edge_responses(image, label, probabilities)
    valid = model._doris_target_valid_mask(label, spatial_valid)
    reliability = ((confidence - model.struct_edge_cfg.get('confidence_floor', 0.60)) /
                   model.struct_edge_cfg.get('confidence_range', 0.35)).clamp(0, 1).unsqueeze(1)
    maps = dict(R_conf=confidence, **{k: structural[k] for k in
                ('R_continuity', 'R_adj', 'R_sem', 'R_struct')})
    maps.update({k: value[:, 0] for k, value in boundary.items()})
    # Canonical paper names; retain old NPZ keys as compatibility aliases.
    maps['R_loc'], maps['R_prob'] = maps['R_continuity'], maps['R_sem']
    maps.update(pseudo_label=label, spatial_valid=valid[:, 0],
                boundary_reliability=reliability[:, 0],
                boundary_loss_weight=(boundary['R_edge'] * reliability * valid *
                    model.struct_edge_loss_weight_tgt * model._struct_edge_warmup_ratio())[:, 0])
    safe = label.clamp(0, model.num_classes - 1)
    maps['adjacency_available'] = model.struct_gate.source_adj_initialized[safe]
    maps['semantic_available'] = model.struct_gate.semantic_proto_initialized[safe]
    return maps


def save_strip(maps, names, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3.5),
                             constrained_layout=True)
    cmap = copy.copy(plt.get_cmap('viridis'))
    cmap.set_bad('lightgray')
    for axis, name in zip(axes, names):
        values = maps[name][0].detach().float().cpu().numpy()
        available_key = {'R_adj': 'adjacency_available',
                         'R_sem': 'semantic_available', 'R_prob': 'semantic_available'}.get(name)
        if available_key is not None:
            available = maps[available_key][0].cpu().numpy()
            values = np.ma.array(values, mask=~available)
        artist = axis.imshow(values, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        axis.set_title(name)
        axis.axis('off')
    fig.colorbar(artist, ax=list(axes), ticks=[0, 0.5, 1], shrink=0.7)
    fig.savefig(str(output), dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--indices', nargs='+', type=int, default=[0],
                        help='one to ten fixed dataset indices; default: 0')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--iteration', type=int,
                        help='override completed iteration if checkpoint metadata lacks it')
    parser.add_argument('--save-npz', action='store_true')
    parser.add_argument('--allow-missing-priors', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= len(args.indices) <= 10 or len(set(args.indices)) != len(args.indices):
        raise ValueError('Choose 1..10 distinct fixed indices.')
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmseg.datasets import build_dataset
    from mmseg.models.builder import build_train_model
    from tools.export_pbcr_edge_debug import get_batch_img, get_batch_img_metas

    cfg = Config.fromfile(args.config)
    if (cfg.uda.struct_cfg.get('fusion_mode') != 'doris_confidence_guided' or
            cfg.uda.struct_edge_cfg.get('edge_fusion_mode') != 'doris_reliable'):
        raise ValueError('Use a DORIS Full config with both new fusion modes enabled.')
    # Checkpoint supplies model weights; never substitute pretrained initialization.
    cfg.model.pretrained = None
    cfg.model.backbone.pretrained = None
    cfg.model.train_cfg.work_dir = args.out_dir
    model = build_train_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    readiness = load_response_checkpoint(model, checkpoint, args.allow_missing_priors)
    iteration = args.iteration if args.iteration is not None else checkpoint.get('meta', {}).get('iter')
    if iteration is None or iteration < 0:
        raise ValueError('Checkpoint iteration missing/invalid; explicitly supply --iteration.')
    model.local_iter = int(iteration)
    device = torch.device(args.device)
    model.to(device).eval()
    dataset_cfg = copy.deepcopy(cfg.data[args.split])
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    if min(args.indices) < 0 or max(args.indices) >= len(dataset):
        raise IndexError('Requested sample index is outside the dataset.')
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(config=osp.abspath(args.config), checkpoint=osp.abspath(args.checkpoint),
                    split=args.split, indices=args.indices, iteration=int(iteration),
                    predictor='EMA Teacher, eval, FP32, generate_pseudo_label',
                    colormap='viridis', display_range=[0, 1], priors=readiness, samples=[])
    margins = cfg.data.train.target.get('crop_pseudo_margins', [0, 0, 0, 0])
    for index in args.indices:
        data = collate([dataset[index]], samples_per_gpu=1)
        if device.type == 'cuda':
            data = scatter(data, [device.index or 0])[0]
        image = get_batch_img(data, device).float()
        metas = get_batch_img_metas(data)
        spatial = image.new_ones((1, 1, 256, 256))
        top, bottom, left, right = margins
        if top:
            spatial[:, :, :top] = 0
        if bottom:
            spatial[:, :, -bottom:] = 0
        if left:
            spatial[:, :, :, :left] = 0
        if right:
            spatial[:, :, :, -right:] = 0
        spatial = F.interpolate(spatial, size=image.shape[-2:], mode='nearest')
        maps = compute_responses(model, image, metas, spatial)
        stem = Path(metas[0]['ori_filename']).stem
        prefix = output / ('{:04d}_{}'.format(index, stem))
        save_strip(maps, ('R_conf', 'R_loc', 'R_adj', 'R_prob', 'R_struct'),
                   str(prefix) + '_mparc.png')
        save_strip(maps, ('E_prob', 'E_img', 'R_pair', 'R_edge'),
                   str(prefix) + '_pbcr.png')
        if args.save_npz:
            np.savez_compressed(str(prefix) + '_responses.npz', **{
                key: value[0].detach().cpu().numpy() for key, value in maps.items()})
        metadata['samples'].append(dict(index=index, filename=metas[0]['ori_filename'],
                                        response_shape=list(image.shape[-2:])))
        print('Exported:', prefix)
    (output / 'response_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
