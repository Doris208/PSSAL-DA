# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
import time
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.uda.phy_reliability import (PhyReliabilityHead,
    HierarchicalPhyReliabilityHead, hierarchical_source_loss,
    hierarchical_phy_gate, phy_gate_parameters, validate_phy_checkpoint, masked_bce_loss, target_phy_loss)
from mmseg.models.uda.proto_gate import ProtoGate
from mmseg.models.uda.struct_prior_gate import StructPriorConsistencyGate
from mmseg.models.losses.struct_edge_loss import label_to_edge
from mmseg.models.utils.dacs_transforms import (convert_to_norm, denorm,
                                                get_class_masks, get_mean_std,
                                                get_vis_image,
                                                strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio
from mmseg.utils.mechanism import training_record


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.output_mode = cfg.get('output_mode', 'minimal')
        if self.output_mode not in ('minimal', 'diagnostic'):
            raise ValueError('output_mode must be minimal or diagnostic')
        self.log_debug_stats = self.output_mode == 'diagnostic'
        self.paper_process_logging = cfg.get('paper_process_logging', False)
        self.doris_process_logging = cfg.get('doris_process_logging', False)
        self.doris_log_interval = cfg.get('doris_log_interval', 50)
        self.mechanism_logging = cfg.get('mechanism_logging', False)
        self.mechanism_cfg = dict(
            enabled=self.mechanism_logging,
            summary_interval=cfg.get('mechanism_summary_interval', 500),
            sample_interval=cfg.get('mechanism_sample_interval', 100),
            probe_enabled=cfg.get('fixed_probe_enabled', False),
            probe_interval=cfg.get('fixed_probe_interval', 500),
            save_raw=cfg.get('mechanism_save_raw', True),
            make_figures=cfg.get('mechanism_make_figures', True))
        self.mechanism_cfg['physiology'] = cfg.get('physiology_collection', False)
        self.mechanism_cfg['output_mode'] = cfg.get('mechanism_output_mode', 'branches')
        self.mechanism_cfg['detailed_output'] = cfg.get('mechanism_detailed_output', False)
        self.mechanism_cfg['online_plot'] = cfg.get('mechanism_online_plot', False)
        self.mechanism_cfg['strict'] = cfg.get('mechanism_strict', True)
        self.mechanism_records = []
        self.save_class_mix_debug = self.log_debug_stats and cfg.get(
            'save_class_mix_debug', False)
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.enable_mixing = cfg.get('use_mixing', True)
        self.save_teacher_pseudo_debug = self.log_debug_stats and cfg.get(
            'save_teacher_pseudo_debug', False)
        self.teacher_pseudo_debug_interval = cfg.get(
            'teacher_pseudo_debug_interval', 1000)
        self.teacher_pseudo_debug_max_samples = cfg.get(
            'teacher_pseudo_debug_max_samples', 5)
        self.teacher_pseudo_debug_dir = cfg.get('teacher_pseudo_debug_dir',
                                                'teacher_pseudo_debug')
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg.get('mask_mode', None)
        self.mask_lambda = cfg.get('mask_lambda', 0.0)
        self.enable_mic = self.mask_mode is not None and self.mask_lambda > 0
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.phy_cfg = cfg.get('phy_cfg', dict(enable=False))
        self.enable_phy = self.phy_cfg.get('enable', False)
        self.struct_cfg = cfg.get('struct_cfg', dict(enable=False))
        self.enable_struct = self.struct_cfg.get('enable', False)
        self.proto_cfg = cfg.get('proto_cfg', dict(enable=False))
        self.enable_proto = self.proto_cfg.get('enable', False)
        self.struct_edge_cfg = cfg.get('struct_edge_cfg', dict(enable=False))
        self.enable_struct_edge = self.struct_edge_cfg.get('enable', False)
        self.joint_gate_cfg = cfg.get('joint_gate_cfg', dict(enable=False))
        self.enable_joint_gate = self.joint_gate_cfg.get('enable', False)
        self.edge_fusion_mode = self.struct_edge_cfg.get('edge_fusion_mode', 'legacy')
        if self.edge_fusion_mode not in ('legacy', 'doris_reliable'):
            raise ValueError('Unknown edge_fusion_mode: ' + self.edge_fusion_mode)
        self.doris_edge = (self.enable_struct_edge and
                           self.edge_fusion_mode == 'doris_reliable')
        if self.struct_edge_cfg.get('pair_response_mode', 'legacy') not in (
                'legacy', 'valid_boundary_only'):
            raise ValueError('Unknown pair_response_mode')
        if self.doris_edge and self.struct_edge_cfg.get(
                'edge_weight_mode', 'struct') != 'paper_pbcr':
            raise ValueError('doris_reliable requires edge_weight_mode=paper_pbcr')
        self.input_channel_order = cfg.get(
            'input_channel_order',
            ['B', 'G', 'R', 'NDVI', 'NDRE', 'CIRE'])
        self.class_names = cfg.get(
            'class_names',
            ['goji', 'vegetation', 'building', 'water', 'road'])
        self.reduced_label_note = cfg.get(
            'reduced_label_note',
            'raw labels 1..5 are reduced to 0..4; ignore=255')
        mmcv.print_log(
            'Ablation mode: '
            f'physiology={self.enable_phy}, '
            f'structural={self.enable_struct}, '
            f'prototype={self.enable_proto}, '
            f'edge={self.enable_struct_edge}',
            'mmseg')
        if 'source_input_channel_order' in cfg:
            for domain in ('source', 'target'):
                mmcv.print_log(domain + ' normalized input order: ' + ', '.join(
                    cfg[domain + '_input_channel_order']), 'mmseg')
        else:
            mmcv.print_log(
                'Input channels/order: ' + ', '.join(self.input_channel_order), 'mmseg')
        mmcv.print_log(
            'Class ids: ' + ', '.join(
                f'{i}={name}' for i, name in enumerate(self.class_names)) +
            f'; {self.reduced_label_note}',
            'mmseg')
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_mic:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        if self.enable_phy:
            phy_head_cfg = deepcopy(self.phy_cfg.get('phy_head', {}))
            self.phy_mode = self.phy_cfg.get('mode', 'legacy_binary')
            if self.phy_mode not in ('legacy_binary', 'hierarchical_gv'):
                raise ValueError('Unknown physiology head mode: ' + self.phy_mode)
            head = (HierarchicalPhyReliabilityHead if self.phy_mode == 'hierarchical_gv'
                    else PhyReliabilityHead)
            self.phy_head = head(**phy_head_cfg)
            self.beta_gv = self.phy_cfg.get('beta_gv', 1.0)
            self.phy_positive_classes = self.phy_cfg.get(
                'phy_positive_classes', [0, 1])
            self.phy_apply_to_classes = self.phy_cfg.get(
                'apply_to_classes', self.phy_positive_classes)
            self.lambda_phy_src = self.phy_cfg.get('lambda_phy_src', 0.1)
            self.lambda_phy_tgt = self.phy_cfg.get('lambda_phy_tgt', 0.0)
            self.phy_target_supervision_mode = self.phy_cfg.get(
                'phy_target_supervision_mode', 'legacy')
            if self.phy_target_supervision_mode not in ('legacy', 'balanced_high_conf'):
                raise ValueError('Unknown phy_target_supervision_mode')
            self.tau_conf_phy = self.phy_cfg.get('tau_conf_phy', self.pseudo_threshold)
            self.phy_pos_weight = self.phy_cfg.get('phy_pos_weight', 0.5)
            self.phy_neg_weight = self.phy_cfg.get('phy_neg_weight', 0.5)
            self.use_target_gate = self.phy_cfg.get('use_target_gate', True)
            self.phy_min_weight = self.phy_cfg.get('phy_min_weight', 0.5)
            self.phy_filter_threshold = self.phy_cfg.get(
                'phy_filter_threshold', None)
            self.phy_filter_mode = self.phy_cfg.get('phy_filter_mode', 'soft')
            self.phy_warmup_iters = self.phy_cfg.get('phy_warmup_iters', 1000)
            self.phy_rampup_iters = self.phy_cfg.get('phy_rampup_iters', 0)
            self.phy_gate_params = phy_gate_parameters(self.phy_cfg)
            self.detach_target_gate = self.phy_cfg.get(
                'detach_target_gate', True)
            if self.phy_mode == 'hierarchical_gv':
                if (self.beta_gv < 0 or not self.detach_target_gate or
                        self.phy_filter_threshold is not None or
                        not 0 < self.phy_min_weight <= 1 or
                        self.phy_cfg.get('lambda_phy_tgt_gv', 0.0) != 0 or
                        self.phy_target_supervision_mode != 'balanced_high_conf' or
                        list(self.phy_positive_classes) != [0, 1] or
                        list(self.phy_apply_to_classes) != [0, 1]):
                    raise ValueError('Invalid H-Phy V1 loss/gate configuration')
            self.save_phy_debug = self.log_debug_stats and self.phy_cfg.get(
                'save_phy_debug', False)
            self.phy_debug_interval = self.phy_cfg.get(
                'phy_debug_interval', 1000)
            self.phy_debug_max_samples = self.phy_cfg.get(
                'phy_debug_max_samples', 2)
            self.phy_debug_dir = self.phy_cfg.get('phy_debug_dir',
                                                  'phy_debug')
        else:
            self.phy_head = None

        if self.enable_struct:
            num_classes = cfg['model']['decode_head']['num_classes']
            struct_gate_cfg = deepcopy(self.struct_cfg)
            struct_gate_cfg.pop('enable', None)
            self.save_struct_debug = self.log_debug_stats and self.struct_cfg.get(
                'save_struct_debug', False)
            struct_gate_cfg.pop('save_struct_debug', None)
            self.struct_debug_interval = struct_gate_cfg.pop(
                'struct_debug_interval', 1000)
            self.struct_debug_max_samples = struct_gate_cfg.pop(
                'struct_debug_max_samples', 2)
            self.struct_debug_dir = struct_gate_cfg.pop(
                'struct_debug_dir', 'struct_gate_debug')
            struct_gate_cfg.pop('debug', None)
            self.struct_gate = StructPriorConsistencyGate(
                num_classes=num_classes, **struct_gate_cfg)
        else:
            self.struct_gate = None

        if self.enable_proto:
            proto_gate_cfg = deepcopy(self.proto_cfg)
            proto_gate_cfg.pop('enable', None)
            self.save_proto_debug = self.log_debug_stats and proto_gate_cfg.get(
                'save_proto_debug', False)
            self.proto_debug_interval = proto_gate_cfg.get(
                'proto_debug_interval', 1500)
            proto_gate_cfg['log_stats'] = (self.log_debug_stats or
                                           self.paper_process_logging)
            proto_gate_cfg['collect_debug'] = self.save_proto_debug
            self.proto_gate = ProtoGate(**proto_gate_cfg)
            if self.doris_process_logging:
                mmcv.print_log(
                    'PBCR prototype selection: tau_proto={} on w_t_ps (after MPARC, '
                    'before PBCR); tau_conf_proto={}; min_pixels={}; '
                    'DACS pseudo_threshold={} is separate.'.format(
                        self.proto_gate.tau_proto, self.proto_gate.tau_conf_proto,
                        self.proto_gate.min_pixels, self.pseudo_threshold), 'mmseg')
        else:
            self.proto_gate = None

        if self.enable_struct_edge:
            self.struct_edge_use_edge_loss = self.struct_edge_cfg.get(
                'use_edge_loss', True)
            self.struct_edge_use_struct_gate = self.struct_edge_cfg.get(
                'use_struct_gate', True)
            self.struct_edge_radius = self.struct_edge_cfg.get(
                'edge_radius', 2)
            self.struct_edge_gate_min = self.struct_edge_cfg.get(
                'gate_min', 0.25)
            self.struct_edge_gate_temp = self.struct_edge_cfg.get(
                'gate_temp', 2.0)
            self.struct_edge_gate_dilate = self.struct_edge_cfg.get(
                'gate_dilate', 3)
            self.struct_edge_loss_weight_src = self.struct_edge_cfg.get(
                'loss_weight_src', 0.02)
            self.struct_edge_loss_weight_tgt = self.struct_edge_cfg.get(
                'loss_weight_tgt', 0.04)
            self.struct_edge_warmup_iters = self.struct_edge_cfg.get(
                'warmup_iters', 3000)
            self.struct_edge_use_img_struct = self.struct_edge_cfg.get(
                'use_img_struct', True)
            self.struct_edge_use_conf_weight = self.struct_edge_cfg.get(
                'use_conf_weight', True)
            self.struct_edge_use_proto_gate = self.struct_edge_cfg.get(
                'use_proto_gate', True)
            self.struct_edge_detach_gate = self.struct_edge_cfg.get(
                'detach_gate', True)
            self.struct_edge_use_spec_gate = self.struct_edge_cfg.get(
                'use_spec_gate', False)
            self.struct_edge_mode = self.struct_edge_cfg.get(
                'edge_weight_mode', 'struct')
            self.struct_edge_prob_weight = self.struct_edge_cfg.get(
                'prob_edge_weight', 0.4)
            self.struct_edge_img_weight = self.struct_edge_cfg.get(
                'img_edge_weight', 0.4)
            self.struct_edge_pair_weight = self.struct_edge_cfg.get(
                'pair_edge_weight', 0.2)
            self.struct_edge_vi_names = self.struct_edge_cfg.get(
                'vi_names', ['ndvi', 'ndre', 'cire'])
            self.struct_edge_vi_channels = self.struct_edge_cfg.get(
                'vi_channels', len(self.struct_edge_vi_names))
            self.struct_edge_spec_fuse = self.struct_edge_cfg.get(
                'spec_fuse', 'max')
            self.struct_edge_spec_weight = self.struct_edge_cfg.get(
                'spec_gate_weight', 0.3)
            self.struct_edge_struct_weight = self.struct_edge_cfg.get(
                'struct_gate_weight', 0.7)
            self.struct_edge_spec_temp = self.struct_edge_cfg.get(
                'spec_gate_temp', 2.0)
            self.struct_edge_spec_dilate = self.struct_edge_cfg.get(
                'spec_gate_dilate', 3)
            self.struct_edge_strict_spec = self.struct_edge_cfg.get(
                'strict_spec', False)
            self.struct_edge_use_pair_prior = self.struct_edge_cfg.get(
                'use_boundary_pair_prior', False)
            self.struct_edge_num_classes = self.struct_edge_cfg.get(
                'num_classes', self.proto_cfg.get('num_classes', 5))
            self.struct_edge_pair_momentum = self.struct_edge_cfg.get(
                'boundary_pair_momentum', 0.99)
            self.struct_edge_pair_min_weight = self.struct_edge_cfg.get(
                'boundary_pair_min_weight', 0.55)
            self.struct_edge_pair_power = self.struct_edge_cfg.get(
                'boundary_pair_power', 0.5)
            self.struct_edge_pair_warmup_iters = self.struct_edge_cfg.get(
                'boundary_pair_warmup_iters', self.struct_edge_warmup_iters)
            self.struct_edge_ignore_index = self.struct_edge_cfg.get(
                'ignore_index', 255)
            self.struct_edge_use_spatial_prior = self.struct_edge_cfg.get(
                'use_spatial_layout_prior', False)
            self.struct_edge_spatial_middle_weight = self.struct_edge_cfg.get(
                'spatial_middle_weight', 1.0)
            self.struct_edge_spatial_edge_weight = self.struct_edge_cfg.get(
                'spatial_edge_weight', 0.9)
            self.struct_edge_spatial_corner_weight = self.struct_edge_cfg.get(
                'spatial_corner_weight', 0.8)
            self.struct_edge_spatial_border_width = self.struct_edge_cfg.get(
                'spatial_border_width', 16)
            self.struct_edge_spatial_border_min_weight = \
                self.struct_edge_cfg.get('spatial_border_min_weight', 0.6)
            self.struct_edge_spatial_stride = self.struct_edge_cfg.get(
                'spatial_stride', 128)
            self.struct_edge_use_neighbor_prior = self.struct_edge_cfg.get(
                'use_neighbor_patch_prior', False)
            self.struct_edge_neighbor_min_weight = self.struct_edge_cfg.get(
                'neighbor_patch_min_weight', 0.70)
            self.struct_edge_neighbor_power = self.struct_edge_cfg.get(
                'neighbor_patch_power', 0.5)
            self.struct_edge_neighbor_momentum = self.struct_edge_cfg.get(
                'neighbor_patch_momentum', 0.90)
            self.struct_edge_neighbor_warmup_iters = self.struct_edge_cfg.get(
                'neighbor_patch_warmup_iters', 3000)
            self.struct_edge_neighbor_max_items = self.struct_edge_cfg.get(
                'neighbor_patch_max_items', 4096)
            self.register_buffer(
                'struct_edge_pair_prior',
                torch.zeros(self.struct_edge_num_classes,
                            self.struct_edge_num_classes))
            self.register_buffer('struct_edge_pair_prior_ready',
                                 torch.zeros(1, dtype=torch.bool))
            self.struct_edge_neighbor_memory = {}
            self.struct_edge_spec_warned = False
            self.struct_edge_proto_warned = False
            self.struct_edge_conf_warned = False

        if self.enable_joint_gate:
            assert self.joint_gate_cfg.get(
                'fusion_mode',
                'classwise_disjoint') == 'classwise_disjoint'
            self.joint_phy_classes = self.joint_gate_cfg.get(
                'phy_classes', [0, 1])
            self.joint_struct_classes = self.joint_gate_cfg.get(
                'struct_classes', [2, 3, 4])
            self.joint_clamp_min = self.joint_gate_cfg.get('clamp_min', 0.0)
            self.joint_clamp_max = self.joint_gate_cfg.get('clamp_max', 1.0)
            self.save_joint_debug = self.log_debug_stats and self.joint_gate_cfg.get(
                'save_joint_debug', False)
            self.joint_debug_interval = self.joint_gate_cfg.get(
                'joint_debug_interval', 1000)
            self.joint_debug_dir = self.joint_gate_cfg.get(
                'joint_debug_dir', 'phy_struct_gate_debug')

        if self.doris_process_logging:
            mmcv.print_log(
                'Paper log: r_loc=R_continuity, r_prob=R_sem; '
                'loss_phy_source_raw is unweighted; loss_phy_source includes lambda_phy_src. '
                'Process fields are sampled snapshots, not accuracy metrics.', 'mmseg')
            if self.enable_struct:
                mmcv.print_log(
                    'MPARC: r_struct=r_conf*(0.40*r_loc+0.30*r_adj+0.30*r_prob); '
                    'uninitialized priors excluded and remaining weights renormalized; '
                    'struct/feature warmup={}, ramp={}.'.format(
                        self.struct_gate.struct_warmup_iters,
                        self.struct_gate.struct_rampup_iters), 'mmseg')
            if self.doris_edge:
                mmcv.print_log(
                    'PBCR: r_edge=0.45*e_prob+0.35*e_img+0.20*r_pair (Teacher target); '
                    'source uses e_img; edge multiplier ramps from 0 to 1 over {} iters; '
                    'lambda_src={}, lambda_tgt={}.'.format(
                        self.struct_edge_warmup_iters, self.struct_edge_loss_weight_src,
                        self.struct_edge_loss_weight_tgt), 'mmseg')

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        observation_start = time.perf_counter() if getattr(self, 'mechanism_logging', False) else None
        log_vars = self(**data_batch)
        optimizer.step()
        # Sum already materialized loss scalars for reporting only. All
        # backward calls and the optimizer update above remain unchanged.
        log_vars['loss'] = sum(
            value * (self.lambda_phy_src if name == 'loss_phy_src' else 1.0)
            for name, value in log_vars.items()
            if 'loss' in name and name != 'loss' and not name.startswith('doris.')
            and name not in ('loss_phy_src_veg_raw', 'loss_phy_src_gv_raw',
                             'loss_phy_src_raw', 'loss_phy_src_weighted', 'loss_phy_tgt_veg_weighted',
                             'loss_phy_tgt_gv_weighted'))
        if observation_start is not None:
            log_vars['train_iter_time'] = time.perf_counter() - observation_start

        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        head = self.get_model().decode_head
        if self.doris_process_logging and self.enable_struct_edge and hasattr(head, 'modules'):
            for module in head.modules():
                if module.__class__.__name__ == 'StructEdgeLoss':
                    module.collect_diagnostics = self._doris_log_enabled()
        if self.save_class_mix_debug and self.debug_img_interval > 0 and \
                self.local_iter % self.debug_img_interval == 0:
            self.get_model().decode_head.debug = True
            self.get_ema_model().decode_head.debug = True
        else:
            self.get_model().decode_head.debug = False
            self.get_ema_model().decode_head.debug = False
        if self.mic is not None:
            self.mic.debug = (self.save_class_mix_debug and
                              self.debug_img_interval > 0 and
                              self.local_iter % self.debug_img_interval == 0)

    def _get_teacher_debug_dir(self):
        debug_dir = self.teacher_pseudo_debug_dir
        if not os.path.isabs(debug_dir):
            debug_dir = os.path.join(self.train_cfg['work_dir'], debug_dir)
        return debug_dir

    def _save_teacher_raw_pseudo_label(self, pseudo_label, out_file):
        palette = getattr(self, 'PALETTE', None)
        if palette is None:
            palette = getattr(self.get_model(), 'PALETTE', None)
        if palette is None:
            palette = getattr(self.get_ema_model(), 'PALETTE', None)
        if palette is None and hasattr(self, 'train_cfg') and \
                self.train_cfg is not None:
            palette = self.train_cfg.get('PALETTE', None)
        if palette is None:
            raise ValueError('PALETTE is required to save teacher labels.')
        palette = np.asarray(palette, dtype=np.uint8)
        color_seg = np.zeros((pseudo_label.shape[0], pseudo_label.shape[1], 3),
                             dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[pseudo_label == label] = color
        Image.fromarray(color_seg, mode='RGB').save(out_file)

    def _save_teacher_confidence_map(self, confidence_map, out_file):
        conf_norm = np.clip(confidence_map, 0.0, 1.0)
        conf_color = (plt.get_cmap('jet')(conf_norm)[..., :3] * 255).astype(
            np.uint8)
        Image.fromarray(conf_color, mode='RGB').save(out_file)

    def _maybe_save_teacher_pseudo_debug(self, pseudo_label, confidence_map):
        if not self.save_teacher_pseudo_debug:
            return
        if self.local_iter % self.teacher_pseudo_debug_interval != 0:
            return
        debug_dir = self._get_teacher_debug_dir()
        label_dir = os.path.join(debug_dir, 'teacher_raw_pseudo_label')
        conf_dir = os.path.join(debug_dir, 'teacher_target_confidence_map')
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(conf_dir, exist_ok=True)
        num_samples = min(pseudo_label.shape[0],
                          self.teacher_pseudo_debug_max_samples)
        for i in range(num_samples):
            filename = f'{(self.local_iter + 1):06d}_{i}.png'
            self._save_teacher_raw_pseudo_label(
                pseudo_label[i].detach().cpu().numpy(),
                os.path.join(label_dir, filename))
            self._save_teacher_confidence_map(
                confidence_map[i].detach().cpu().numpy(),
                os.path.join(conf_dir, filename))

    def _class_mask(self, label, classes):
        class_tensor = torch.tensor(classes, device=label.device)
        return torch.any(label[..., None] == class_tensor, dim=-1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.enable_phy and (self.phy_mode == 'hierarchical_gv' or
                               any(key.startswith((prefix + 'phy_head.encoder.',
                                                   prefix + 'phy_head.veg_head.',
                                                   prefix + 'phy_head.gv_head.'))
                                   for key in state_dict)):
            validate_phy_checkpoint(self.phy_mode, state_dict, prefix + 'phy_head.')
            # MMCV's non-strict loader otherwise only warns on incomplete heads.
            for key, value in self.phy_head.state_dict().items():
                saved = state_dict.get(prefix + 'phy_head.' + key)
                if saved is None or saved.shape != value.shape:
                    raise RuntimeError('Incomplete physiology checkpoint tensor: ' + key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _compute_phy_src_loss(self, source_vi_img, gt_semantic_seg, log_vars):
        r_phy_src = self.phy_head(source_vi_img.float())
        if self.phy_mode == 'hierarchical_gv':
            raw, parts = hierarchical_source_loss(r_phy_src, gt_semantic_seg, self.beta_gv)
            log_vars.update(loss_phy_src_veg_raw=parts['veg'].detach().item(),
                            loss_phy_src_gv_raw=parts['gv'].detach().item(),
                            loss_phy_src_raw=raw.detach().item(),
                            loss_phy_src_weighted=(self.lambda_phy_src * raw).detach().item())
            log_vars['loss_phy_src'] = log_vars['loss_phy_src_raw']
            for name in ('valid_count', 'goji_count', 'other_veg_count'):
                log_vars['hphy_src_' + name] = parts[name].item()
            return r_phy_src, self.lambda_phy_src * raw
        gt = gt_semantic_seg.squeeze(1)
        target = self._class_mask(gt, self.phy_positive_classes).unsqueeze(1)
        valid = gt.ne(255).unsqueeze(1)
        loss_phy_src = masked_bce_loss(r_phy_src, target, valid)
        if self._mechanism_sample_enabled() and (
                self.mechanism_cfg['output_mode'] != 'branches' or self.mechanism_cfg['detailed_output']):
            with torch.no_grad():
                self.mechanism_phy_source = {
                    'vegetation': (r_phy_src.detach()[valid & target].float().sum().item(),
                                   int((valid & target).sum())),
                    'nonvegetation': (r_phy_src.detach()[valid & ~target].float().sum().item(),
                                      int((valid & ~target).sum()))}
        log_vars['loss_phy_src'] = loss_phy_src.item()
        if self.doris_process_logging:
            log_vars['doris.phy_source_weighted'] = self.lambda_phy_src * log_vars['loss_phy_src']
        if self.log_debug_stats:
            log_vars['phy_src_mean'] = r_phy_src.detach().mean().item()
        return r_phy_src, self.lambda_phy_src * loss_phy_src

    def _apply_phy_gate(self, target_vi_img, pseudo_label, pseudo_weight,
                        log_vars, valid_mask=None):
        r_phy_tgt = self.phy_head(target_vi_img.float())
        if self.phy_mode == 'hierarchical_gv':
            if r_phy_tgt['r_veg'].shape[-2:] != pseudo_label.shape[-2:]:
                raise ValueError('H-Phy VI and teacher labels must share the pixel grid')
            details = hierarchical_phy_gate(
                r_phy_tgt, pseudo_label, pseudo_weight, self.local_iter,
                min_weight=self.phy_min_weight, enabled=self.use_target_gate,
                valid_mask=valid_mask, return_details=True, **self.phy_gate_params)
            after, gate, agreement = details['W_after'], details['G_phy'], details['S_phy']
            r_phy_tgt.update(s_phy_class_agreement=agreement.unsqueeze(1),
                             g_phy=gate.unsqueeze(1), a_phy=details['A_phy'].unsqueeze(1),
                             w_before_phy=details['W_before'].detach().clone().unsqueeze(1),
                             w_after_phy=after.detach().clone().unsqueeze(1))
            spatial = (pseudo_weight.gt(0) if valid_mask is None
                       else valid_mask.reshape_as(pseudo_label).gt(0))
            valid = spatial & pseudo_label.ge(0) & pseudo_label.lt(self.num_classes)
            for cls, name in ((0, 'goji'), (1, 'vegetation')):
                mask = valid & pseudo_label.eq(cls)
                log_vars['hphy_' + name + '_pseudo_ratio'] = (
                    mask.sum().float() / valid.sum().clamp_min(1)).item()
                for key, value in (('agreement', agreement), ('gate', gate)):
                    log_vars['hphy_' + name + '_' + key + '_mean'] = (
                        value[mask].mean().item() if mask.any() else 0.0)
            supported = valid & pseudo_weight.gt(0)
            active = supported & self._class_mask(pseudo_label, [0, 1]) & gate.lt(1)
            log_vars.update(phy_gate_iter=self.local_iter, phy_gate_rho=details['rho'],
                phy_gate_active_ratio=(active.sum().float() / supported.sum().clamp_min(1)).item(),
                phy_S_mean_active=agreement[active].mean().item() if active.any() else 0.0,
                phy_A_mean_active=details['A_phy'][active].mean().item() if active.any() else 0.0,
                phy_A_max_active=details['A_phy'][active].max().item() if active.any() else 0.0)
            log_vars['hphy_attenuation_mean_valid_weight'] = (
                (1 - gate)[supported].mean().item() if supported.any() else 0.0)
            return r_phy_tgt, after, gate
        if r_phy_tgt.shape[-2:] != pseudo_label.shape[-2:]:
            r_phy_tgt = F.interpolate(
                r_phy_tgt,
                size=pseudo_label.shape[-2:],
                mode='bilinear',
                align_corners=False)
        if self.log_debug_stats:
            log_vars['phy_tgt_mean'] = r_phy_tgt.detach().mean().item()
        if not self.use_target_gate or self.local_iter < self.phy_warmup_iters:
            return r_phy_tgt, pseudo_weight, None

        gate_src = r_phy_tgt.detach() if self.detach_target_gate else r_phy_tgt
        gate_src = gate_src.squeeze(1)
        gate = self.phy_min_weight + (1 - self.phy_min_weight) * gate_src
        if self.phy_filter_threshold is not None:
            unreliable = gate_src.lt(self.phy_filter_threshold)
            if self.phy_filter_mode == 'hard':
                gate = torch.where(unreliable, torch.zeros_like(gate), gate)
            else:
                gate = torch.where(unreliable, gate * 0.5, gate)
        if self.phy_rampup_iters > 0:
            ramp = min(1.0, max(0.0, (
                self.local_iter - self.phy_warmup_iters) /
                float(self.phy_rampup_iters)))
            gate = (1 - ramp) * torch.ones_like(gate) + ramp * gate
        phy_weight = torch.ones_like(pseudo_weight)
        veg_mask = self._class_mask(pseudo_label, self.phy_apply_to_classes)
        phy_weight[veg_mask] = gate[veg_mask].to(dtype=phy_weight.dtype)
        pseudo_weight = pseudo_weight * phy_weight

        if self.log_debug_stats:
            if veg_mask.any():
                log_vars['phy_gate_mean'] = phy_weight[veg_mask].detach().mean().item()
                log_vars['phy_gate_min'] = phy_weight[veg_mask].detach().min().item()
                log_vars['phy_gate_max'] = phy_weight[veg_mask].detach().max().item()
                log_vars['phy_retain_ratio'] = (
                    pseudo_weight[veg_mask] > 0).float().mean().item()
                if self.phy_filter_threshold is not None:
                    log_vars['phy_filter_ratio'] = (
                        gate_src[veg_mask].lt(self.phy_filter_threshold)
                        .float().mean().item())
                else:
                    log_vars['phy_filter_ratio'] = 0.0
            else:
                log_vars['phy_gate_mean'] = 1.0
                log_vars['phy_gate_min'] = 1.0
                log_vars['phy_gate_max'] = 1.0
                log_vars['phy_retain_ratio'] = 1.0
                log_vars['phy_filter_ratio'] = 0.0
            goji_mask = pseudo_label == 0
            veg_cls_mask = pseudo_label == 1
            log_vars['phy_gate_goji_mean'] = (
                phy_weight[goji_mask].detach().mean().item()
                if goji_mask.any() else 1.0)
            log_vars['phy_gate_general_vegetation_mean'] = (
                phy_weight[veg_cls_mask].detach().mean().item()
                if veg_cls_mask.any() else 1.0)
            log_vars['phy_gate_vegetation_mean'] = log_vars['phy_gate_mean']
            log_vars['phy_veg_pseudo_ratio'] = veg_mask.float().mean().item()
        return r_phy_tgt, pseudo_weight, phy_weight

    def _save_phy_map(self, data, out_file, cmap='jet'):
        data = data.detach().float().cpu().numpy()
        data = np.clip(data, 0.0, 1.0)
        color = (plt.get_cmap(cmap)(data)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(color, mode='RGB').save(out_file)

    def _maybe_save_phy_debug(self, img, img_metas, source_vi_img,
                              gt_semantic_seg, target_vi_img, pseudo_label,
                              confidence_map, r_phy_tgt, pseudo_weight,
                              target_img, target_img_metas):
        if not self.enable_phy or not self.save_phy_debug:
            return
        if self.local_iter % self.phy_debug_interval != 0:
            return
        debug_dir = self.phy_debug_dir
        if not os.path.isabs(debug_dir):
            debug_dir = os.path.join(self.train_cfg['work_dir'], debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
        dev = img.device
        src_vis = get_vis_image(img, img_metas, dev)
        with torch.no_grad():
            student_pred = torch.argmax(
                self.get_model().generate_pseudo_label(
                    target_img, target_img_metas),
                dim=1)
        num_samples = min(img.shape[0], self.phy_debug_max_samples)
        for i in range(num_samples):
            prefix = os.path.join(
                debug_dir, f'iter_{(self.local_iter + 1):07d}_sample_{i}')
            Image.fromarray(
                (src_vis[i].detach().cpu().numpy().transpose(1, 2, 0) * 255)
                .clip(0, 255).astype(np.uint8)).save(prefix + '_input.png')
            for c, name in enumerate(self.phy_cfg.get('vi_names',
                                                      ['ndvi', 'ndre',
                                                       'cire'])):
                if c < target_vi_img.shape[1]:
                    self._save_phy_map(target_vi_img[i, c],
                                       prefix + f'_{name}.png')
            self._save_teacher_raw_pseudo_label(
                gt_semantic_seg[i, 0].detach().cpu().numpy(),
                prefix + '_gt.png')
            self._save_teacher_raw_pseudo_label(
                pseudo_label[i].detach().cpu().numpy(),
                prefix + '_pseudo.png')
            self._save_teacher_raw_pseudo_label(
                student_pred[i].detach().cpu().numpy(),
                prefix + '_student.png')
            self._save_teacher_confidence_map(
                confidence_map[i].detach().cpu().numpy(),
                prefix + '_conf.png')
            if isinstance(r_phy_tgt, dict):
                for name, value in r_phy_tgt.items():
                    if not name.startswith('logit_'):
                        self._save_phy_map(value[i, 0], prefix + '_' + name + '.png')
            else:
                self._save_phy_map(r_phy_tgt[i, 0], prefix + '_r_phy.png')
            self._save_phy_map(pseudo_weight[i], prefix + '_weight.png')

    def _log_struct_debug(self, struct_debug, log_vars):
        if not self.log_debug_stats:
            return
        scalar_keys = [
            'struct_gate_mean', 'struct_gate_struct_class_mean',
            'struct_gate_building_mean', 'struct_gate_water_mean',
            'struct_gate_road_mean', 'struct_retain_ratio',
            'struct_filter_ratio', 'struct_continuity_mean',
            'struct_adj_mean', 'struct_sem_mean', 'source_adj_entropy',
            'target_struct_pixel_ratio'
        ]
        for key in scalar_keys:
            if key in struct_debug:
                log_vars[key] = struct_debug[key].detach().item()

    def _log_proto_debug(self, proto_debug, log_vars):
        if not self.log_debug_stats:
            return
        scalar_keys = [
            'proto_weight_mean', 'proto_margin_mean', 'proto_valid_classes'
        ]
        for key in scalar_keys:
            if key in proto_debug:
                log_vars[key] = proto_debug[key].detach().item()
        for cls in range(self.proto_cfg.get('num_classes', 5)):
            for name in ('proto_weight_class', 'proto_margin_class'):
                key = f'{name}_{cls}_mean'
                if key in proto_debug:
                    log_vars[key] = proto_debug[key].detach().item()

    def _log_proto_update(self, prefix, proto_log, log_vars):
        if not self.log_debug_stats and not (
                self.paper_process_logging and prefix == 'proto_tgt'):
            return
        if proto_log is None:
            return
        if 'update_ratio' in proto_log:
            log_vars[f'{prefix}_update_ratio'] = \
                proto_log['update_ratio'].detach().item()
        classes = range(self.proto_cfg.get('num_classes', 5)) \
            if self.log_debug_stats else ()
        for cls in classes:
            key = f'update_class_{cls}_ratio'
            if key in proto_log:
                log_vars[f'{prefix}_update_class_{cls}_ratio'] = \
                    proto_log[key].detach().item()

    def _maybe_save_proto_debug(self, proto_debug, final_pseudo_weight):
        if not self.enable_proto or not self.save_proto_debug:
            return
        if self.local_iter % self.proto_debug_interval != 0:
            return
        debug_dir = self.proto_cfg.get('proto_debug_dir',
                                       'proto_gate_debug')
        if not os.path.isabs(debug_dir):
            debug_dir = os.path.join(self.train_cfg['work_dir'], debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
        proto_weight = proto_debug.get('proto_weight')
        proto_margin = proto_debug.get('proto_margin')
        if proto_weight is None or proto_margin is None:
            return
        num_samples = min(proto_weight.shape[0],
                          self.proto_cfg.get('proto_debug_max_samples', 50))
        for i in range(num_samples):
            prefix = os.path.join(
                debug_dir, f'iter_{(self.local_iter + 1):07d}_sample_{i}')
            self._save_phy_map(proto_weight[i],
                               prefix + '_proto_weight.png')
            self._save_phy_map(proto_margin[i].clamp_min(0.0),
                               prefix + '_proto_margin.png')
            self._save_phy_map(final_pseudo_weight[i],
                               prefix + '_final_pseudo_weight.png')

    def _apply_struct_gate(self, ema_softmax, pseudo_label, pseudo_weight,
                           gt_semantic_seg, log_vars, source_probs=None,
                           target_confidence=None):
        pseudo_weight, struct_debug = self.struct_gate.apply(
            target_probs=ema_softmax,
            pseudo_label=pseudo_label,
            pseudo_weight=pseudo_weight,
            source_gt=gt_semantic_seg,
            source_probs=source_probs,
            iter=self.local_iter,
            ignore_index=255,
            collect_debug=(self.save_struct_debug or
                           (self.enable_joint_gate and self.save_joint_debug)),
            log_stats=self.log_debug_stats,
            target_confidence=target_confidence)
        self._log_struct_debug(struct_debug, log_vars)
        return pseudo_weight, struct_debug

    def _fuse_phy_struct_pseudo_weight(self, original_pseudo_weight,
                                       phy_pseudo_weight,
                                       struct_pseudo_weight, pseudo_label,
                                       log_vars):
        joint_weight = original_pseudo_weight.clone()
        phy_mask = self._class_mask(pseudo_label, self.joint_phy_classes)
        struct_mask = self._class_mask(pseudo_label,
                                       self.joint_struct_classes)
        phy_struct_overlap = phy_mask & struct_mask
        if phy_struct_overlap.any():
            struct_mask = struct_mask & ~phy_mask
        joint_weight[phy_mask] = phy_pseudo_weight[phy_mask]
        joint_weight[struct_mask] = struct_pseudo_weight[struct_mask]
        joint_weight = joint_weight.clamp(
            min=self.joint_clamp_min, max=self.joint_clamp_max)

        if not self.log_debug_stats and not self.save_joint_debug:
            return joint_weight, {}

        changed = (joint_weight - original_pseudo_weight).abs().gt(1e-6)
        filtered = original_pseudo_weight.gt(0) & joint_weight.le(0)
        log_vars['joint_gate_mean'] = joint_weight.detach().mean().item()
        log_vars['joint_retain_ratio'] = (
            joint_weight > 0).float().mean().item()
        log_vars['joint_filter_ratio'] = filtered.float().mean().item()
        log_vars['joint_phy_pixel_ratio'] = phy_mask.float().mean().item()
        log_vars['joint_struct_pixel_ratio'] = struct_mask.float().mean().item()
        for cls, name in [(0, 'goji'), (1, 'vegetation'),
                          (2, 'building'), (3, 'water'), (4, 'road')]:
            cls_mask = pseudo_label == cls
            log_vars[f'joint_{name}_retain_ratio'] = (
                (joint_weight[cls_mask] > 0).float().mean().item()
                if cls_mask.any() else 1.0)
        debug = dict(
            joint_weight=joint_weight.detach(),
            phy_mask=phy_mask.detach(),
            struct_mask=struct_mask.detach(),
            changed_mask=changed.detach())
        return joint_weight, debug

    def _maybe_save_joint_debug(self, target_img, target_img_metas,
                                target_vi_img, pseudo_label, confidence_map,
                                r_phy_tgt, struct_debug,
                                original_pseudo_weight, phy_pseudo_weight,
                                struct_pseudo_weight, joint_pseudo_weight):
        if not self.enable_joint_gate or not self.save_joint_debug:
            return
        if self.local_iter % self.joint_debug_interval != 0:
            return
        debug_dir = self.joint_debug_dir
        if not os.path.isabs(debug_dir):
            debug_dir = os.path.join(self.train_cfg['work_dir'], debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
        dev = target_img.device
        target_vis = get_vis_image(target_img, target_img_metas, dev)
        with torch.no_grad():
            student_pred = torch.argmax(
                self.get_model().generate_pseudo_label(
                    target_img, target_img_metas),
                dim=1)
        num_samples = min(target_img.shape[0],
                          self.joint_gate_cfg.get('joint_debug_max_samples',
                                                  50))
        for i in range(num_samples):
            prefix = os.path.join(
                debug_dir, f'iter_{(self.local_iter + 1):07d}_sample_{i}')
            Image.fromarray(
                (target_vis[i].detach().cpu().numpy().transpose(1, 2, 0) *
                 255).clip(0, 255).astype(np.uint8)).save(
                     prefix + '_input.png')
            for c, name in enumerate(self.phy_cfg.get('vi_names',
                                                      ['ndvi', 'ndre',
                                                       'cire'])):
                if target_vi_img is not None and c < target_vi_img.shape[1]:
                    self._save_phy_map(target_vi_img[i, c],
                                       prefix + f'_{name}.png')
            self._save_teacher_raw_pseudo_label(
                pseudo_label[i].detach().cpu().numpy(),
                prefix + '_target_teacher_pseudo_label.png')
            self._save_teacher_raw_pseudo_label(
                student_pred[i].detach().cpu().numpy(),
                prefix + '_target_student_pred.png')
            self._save_teacher_confidence_map(
                confidence_map[i].detach().cpu().numpy(),
                prefix + '_target_teacher_confidence.png')
            if r_phy_tgt is not None:
                response = (r_phy_tgt['s_phy_class_agreement']
                            if isinstance(r_phy_tgt, dict) else r_phy_tgt)
                self._save_phy_map(response[i, 0],
                                   prefix + ('_target_S_phy.png'
                                             if isinstance(r_phy_tgt, dict) else '_target_R_phy.png'))
            if struct_debug is not None:
                self._save_phy_map(struct_debug['R_struct'][i],
                                   prefix + '_target_R_struct.png')
                self._save_phy_map(struct_debug['R_continuity'][i],
                                   prefix + '_target_R_continuity.png')
                self._save_phy_map(struct_debug['R_adj'][i],
                                   prefix + '_target_R_adj.png')
                self._save_phy_map(struct_debug['R_sem'][i],
                                   prefix + '_target_R_sem.png')
            self._save_phy_map(original_pseudo_weight[i],
                               prefix + '_target_original_pseudo_weight.png')
            self._save_phy_map(phy_pseudo_weight[i],
                               prefix + '_target_phy_pseudo_weight.png')
            self._save_phy_map(struct_pseudo_weight[i],
                               prefix + '_target_struct_pseudo_weight.png')
            self._save_phy_map(joint_pseudo_weight[i],
                               prefix + '_target_joint_pseudo_weight.png')
        if struct_debug is not None:
            self._save_struct_matrix(
                struct_debug['source_adj_matrix'],
                os.path.join(debug_dir, 'source_adj_matrix.png'))
            self._save_struct_matrix(
                struct_debug['target_adj_matrix'],
                os.path.join(debug_dir, 'target_adj_matrix.png'))

    def _save_struct_matrix(self, matrix, out_file):
        matrix = matrix.detach().float().cpu().numpy()
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(matrix, vmin=0, vmax=max(float(matrix.max()), 1e-6),
                       cmap='viridis')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_file)
        plt.close(fig)

    def _maybe_save_struct_debug(self, target_img, target_img_metas,
                                 gt_semantic_seg, pseudo_label,
                                 confidence_map, orig_pseudo_weight,
                                 final_pseudo_weight, struct_debug):
        if not self.enable_struct or not self.save_struct_debug:
            return
        if self.local_iter % self.struct_debug_interval != 0:
            return
        debug_dir = self.struct_debug_dir
        if not os.path.isabs(debug_dir):
            debug_dir = os.path.join(self.train_cfg['work_dir'], debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
        dev = target_img.device
        target_vis = get_vis_image(target_img, target_img_metas, dev)
        with torch.no_grad():
            student_pred = torch.argmax(
                self.get_model().generate_pseudo_label(
                    target_img, target_img_metas),
                dim=1)
        num_samples = min(target_img.shape[0],
                          self.struct_debug_max_samples)
        for i in range(num_samples):
            prefix = os.path.join(
                debug_dir, f'iter_{(self.local_iter + 1):07d}_sample_{i}')
            Image.fromarray(
                (target_vis[i].detach().cpu().numpy().transpose(1, 2, 0) *
                 255).clip(0, 255).astype(np.uint8)).save(
                     prefix + '_input.png')
            self._save_teacher_raw_pseudo_label(
                gt_semantic_seg[i, 0].detach().cpu().numpy(),
                prefix + '_source_gt.png')
            self._save_teacher_raw_pseudo_label(
                pseudo_label[i].detach().cpu().numpy(),
                prefix + '_teacher_pseudo_label.png')
            self._save_teacher_raw_pseudo_label(
                student_pred[i].detach().cpu().numpy(),
                prefix + '_student_prediction.png')
            self._save_teacher_confidence_map(
                confidence_map[i].detach().cpu().numpy(),
                prefix + '_teacher_confidence.png')
            self._save_phy_map(struct_debug['R_continuity'][i],
                               prefix + '_target_R_continuity.png')
            self._save_phy_map(struct_debug['R_adj'][i],
                               prefix + '_target_R_adj.png')
            self._save_phy_map(struct_debug['R_sem'][i],
                               prefix + '_target_R_sem.png')
            self._save_phy_map(struct_debug['R_struct'][i],
                               prefix + '_target_R_struct.png')
            self._save_phy_map(orig_pseudo_weight[i],
                               prefix + '_target_original_pseudo_weight.png')
            self._save_phy_map(final_pseudo_weight[i],
                               prefix +
                               '_target_final_pseudo_weight_struct.png')
        self._save_struct_matrix(struct_debug['source_adj_matrix'],
                                 os.path.join(debug_dir,
                                              'source_adj_matrix.png'))
        self._save_struct_matrix(struct_debug['target_adj_matrix'],
                                 os.path.join(debug_dir,
                                              'target_adj_matrix.png'))

    def _struct_edge_warmup_ratio(self):
        if not self.enable_struct_edge:
            return 0.0
        if self.struct_edge_warmup_iters <= 0:
            return 1.0
        return min(1.0, float(self.local_iter) /
                   float(self.struct_edge_warmup_iters))

    def _build_img_struct_gate(self, img):
        if not self.struct_edge_use_img_struct:
            return img.new_ones((img.shape[0], 1, img.shape[2], img.shape[3]))
        eps = 1e-6
        x = img[:, :3].float() if img.shape[1] > 3 else img.float()
        x_gray = x.mean(dim=1, keepdim=True)
        x_min = x_gray.amin(dim=(2, 3), keepdim=True)
        x_max = x_gray.amax(dim=(2, 3), keepdim=True)
        x_gray = (x_gray - x_min) / (x_max - x_min + eps)
        sobel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gx = F.conv2d(x_gray, sobel_x.view(1, 1, 3, 3), padding=1)
        gy = F.conv2d(x_gray, sobel_y.view(1, 1, 3, 3), padding=1)
        grad = torch.sqrt(gx.pow(2) + gy.pow(2) + eps)
        grad = grad / grad.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
        mean = grad.mean(dim=(2, 3), keepdim=True)
        std = grad.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        gate = torch.sigmoid((grad - mean) / std *
                             self.struct_edge_gate_temp)
        if self.struct_edge_gate_dilate > 1:
            gate = F.max_pool2d(
                gate,
                kernel_size=self.struct_edge_gate_dilate,
                stride=1,
                padding=self.struct_edge_gate_dilate // 2)
        gate = self.struct_edge_gate_min + (
            1.0 - self.struct_edge_gate_min) * gate
        return gate.detach() if self.struct_edge_detach_gate else gate

    def _build_img_edge_response(self, img):
        eps = 1e-6
        x = (img[:, :3] if self.doris_edge else img).float()
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
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + eps).mean(
            dim=1, keepdim=True)
        norm = edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
        return (edge / norm).clamp(0.0, 1.0)

    def _build_prob_edge_response(self, probs):
        eps = 1e-6
        channels = probs.shape[1]
        sobel_x = probs.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = probs.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        gx = F.conv2d(probs.float(), sobel_x, padding=1, groups=channels)
        gy = F.conv2d(probs.float(), sobel_y, padding=1, groups=channels)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + eps).mean(
            dim=1, keepdim=True)
        norm = edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
        return (edge / norm).clamp(0.0, 1.0)

    def _get_vi_tensor(self, img, batch=None, img_metas=None, explicit_vi=None):
        del img_metas
        if explicit_vi is not None:
            return explicit_vi
        if batch is not None:
            for key in ('vi_img', 'vi', 'img_vi', 'vimg'):
                if key in batch:
                    return batch[key]
            vi_parts = [batch.get(key) for key in ('ndvi', 'ndre', 'cire')]
            if all(part is not None for part in vi_parts):
                return torch.cat([
                    part if part.dim() == 4 else part.unsqueeze(1)
                    for part in vi_parts
                ], dim=1)
        vi_channels = self.struct_edge_vi_channels
        if img is not None and img.shape[1] > 3 and vi_channels > 0 and \
                img.shape[1] >= vi_channels:
            return img[:, -vi_channels:, :, :]
        if self.struct_edge_use_spec_gate and not self.struct_edge_spec_warned:
            msg = '[SG-BDE] spectral gate enabled but VI tensor not found, ' \
                  'fallback to struct-only gate.'
            if self.struct_edge_strict_spec:
                raise ValueError(msg)
            mmcv.print_log(msg, 'mmseg')
            self.struct_edge_spec_warned = True
        return None

    def _build_spec_gate(self, vi):
        if vi is None:
            return None
        eps = 1e-6
        x = vi.float()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + eps)
        channels = x.shape[1]
        sobel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        gx = F.conv2d(x, sobel_x, padding=1, groups=channels)
        gy = F.conv2d(x, sobel_y, padding=1, groups=channels)
        grad = torch.sqrt(gx.pow(2) + gy.pow(2) + eps)
        if self.struct_edge_spec_fuse == 'mean':
            grad = grad.mean(dim=1, keepdim=True)
        else:
            grad = grad.max(dim=1, keepdim=True)[0]
        grad = grad / grad.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
        mean = grad.mean(dim=(2, 3), keepdim=True)
        std = grad.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        gate = torch.sigmoid((grad - mean) / std *
                             self.struct_edge_spec_temp)
        if self.struct_edge_spec_dilate > 1:
            gate = F.max_pool2d(
                gate,
                kernel_size=self.struct_edge_spec_dilate,
                stride=1,
                padding=self.struct_edge_spec_dilate // 2)
        gate = self.struct_edge_gate_min + (
            1.0 - self.struct_edge_gate_min) * gate
        return gate.detach() if self.struct_edge_detach_gate else gate

    def _fuse_struct_spec_gate(self, struct_gate, spec_gate, ref_tensor=None):
        if struct_gate is None and spec_gate is None:
            if ref_tensor is None:
                return None
            return ref_tensor.new_ones((ref_tensor.shape[0], 1,
                                        ref_tensor.shape[2],
                                        ref_tensor.shape[3]))
        if struct_gate is None:
            return spec_gate
        if spec_gate is None:
            return struct_gate
        if spec_gate.shape[-2:] != struct_gate.shape[-2:]:
            spec_gate = F.interpolate(
                spec_gate,
                size=struct_gate.shape[-2:],
                mode='bilinear',
                align_corners=False)
        sw = self.struct_edge_struct_weight
        vw = self.struct_edge_spec_weight
        denom = max(sw + vw, 1e-6)
        gate = (sw * struct_gate + vw * spec_gate) / denom
        return gate.clamp(min=self.struct_edge_gate_min, max=1.0)

    def _apply_target_reliability_gate(self,
                                       edge_gate,
                                       pseudo_weight=None,
                                       proto_gate=None):
        if edge_gate is None:
            return None
        if self.struct_edge_use_conf_weight:
            if pseudo_weight is not None:
                if pseudo_weight.dim() == 3:
                    pseudo_weight = pseudo_weight.unsqueeze(1)
                if pseudo_weight.shape[-2:] != edge_gate.shape[-2:]:
                    pseudo_weight = F.interpolate(
                        pseudo_weight.float(),
                        size=edge_gate.shape[-2:],
                        mode='nearest')
                edge_gate = edge_gate * pseudo_weight.detach()
            elif not self.struct_edge_conf_warned:
                mmcv.print_log(
                    '[SG-BDE] confidence gate requested but pseudo_weight '
                    'is missing; skip confidence weighting.', 'mmseg')
                self.struct_edge_conf_warned = True
        if self.struct_edge_use_proto_gate:
            if proto_gate is not None:
                if proto_gate.dim() == 3:
                    proto_gate = proto_gate.unsqueeze(1)
                if proto_gate.shape[-2:] != edge_gate.shape[-2:]:
                    proto_gate = F.interpolate(
                        proto_gate.float(),
                        size=edge_gate.shape[-2:],
                        mode='bilinear',
                        align_corners=False)
                edge_gate = edge_gate * proto_gate.detach()
            elif not self.struct_edge_proto_warned:
                mmcv.print_log(
                    '[SG-BDE] ProtoGate weighting requested but proto gate '
                    'is missing; skip proto weighting.', 'mmseg')
                self.struct_edge_proto_warned = True
        return edge_gate

    def _valid_struct_edge_label(self, label):
        return (label != self.struct_edge_ignore_index) & (label >= 0) & (
            label < self.struct_edge_num_classes)

    def _as_struct_edge_label(self, label):
        if label.dim() == 4:
            label = label[:, 0]
        return label.long()

    def _boundary_pair_matrix(self, label):
        label = self._as_struct_edge_label(label)
        matrix = label.new_zeros(
            (self.struct_edge_num_classes, self.struct_edge_num_classes),
            dtype=torch.float)
        shifts = ((0, 1), (1, 0))
        valid = self._valid_struct_edge_label(label)
        for dy, dx in shifts:
            a = label[:, :-dy or None, :-dx or None]
            b = label[:, dy:, dx:]
            va = valid[:, :-dy or None, :-dx or None]
            vb = valid[:, dy:, dx:]
            boundary = va & vb & a.ne(b)
            if not boundary.any():
                continue
            pair = a[boundary] * self.struct_edge_num_classes + b[boundary]
            counts = torch.bincount(
                pair,
                minlength=self.struct_edge_num_classes *
                self.struct_edge_num_classes).float()
            counts = counts.reshape(self.struct_edge_num_classes,
                                    self.struct_edge_num_classes)
            matrix = matrix + counts + counts.t()
        max_val = matrix.max()
        if max_val > 0:
            matrix = matrix / max_val
        return matrix

    @torch.no_grad()
    def _update_boundary_pair_prior(self, source_label):
        if not self.struct_edge_use_pair_prior:
            return
        pair_matrix = self._boundary_pair_matrix(source_label).to(
            self.struct_edge_pair_prior)
        if pair_matrix.max() <= 0:
            return
        if self.struct_edge_pair_prior_ready.item():
            self.struct_edge_pair_prior.mul_(self.struct_edge_pair_momentum)
            self.struct_edge_pair_prior.add_(
                pair_matrix * (1.0 - self.struct_edge_pair_momentum))
        else:
            self.struct_edge_pair_prior.copy_(pair_matrix)
            self.struct_edge_pair_prior_ready.fill_(True)

    def _build_boundary_pair_gate(self, label):
        label = self._as_struct_edge_label(label)
        gate = label.new_ones(label.shape, dtype=torch.float)
        if not self.struct_edge_use_pair_prior or \
                not self.struct_edge_pair_prior_ready.item():
            return gate.unsqueeze(1)
        pair_prior = self.struct_edge_pair_prior.to(device=label.device,
                                                    dtype=torch.float)
        valid = self._valid_struct_edge_label(label)
        pair_score = label.new_zeros(label.shape, dtype=torch.float)
        has_pair = torch.zeros_like(valid)
        shifts = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for dy, dx in shifts:
            shifted = torch.roll(label, shifts=(dy, dx), dims=(1, 2))
            shifted_valid = torch.roll(valid, shifts=(dy, dx), dims=(1, 2))
            if dy > 0:
                shifted_valid[:, :dy, :] = False
            elif dy < 0:
                shifted_valid[:, dy:, :] = False
            if dx > 0:
                shifted_valid[:, :, :dx] = False
            elif dx < 0:
                shifted_valid[:, :, dx:] = False
            boundary = valid & shifted_valid & label.ne(shifted)
            if not boundary.any():
                continue
            score = pair_prior[label.clamp(0, self.struct_edge_num_classes - 1),
                               shifted.clamp(0, self.struct_edge_num_classes - 1)]
            pair_score = torch.maximum(pair_score,
                                       torch.where(boundary, score,
                                                   pair_score.new_zeros(())))
            has_pair |= boundary
        pair_score = pair_score.clamp(0.0, 1.0).pow(
            self.struct_edge_pair_power)
        pair_gate = self.struct_edge_pair_min_weight + (
            1.0 - self.struct_edge_pair_min_weight) * pair_score
        gate = torch.where(has_pair, pair_gate, gate)
        if self.local_iter < self.struct_edge_pair_warmup_iters:
            ratio = float(self.local_iter) / max(
                float(self.struct_edge_pair_warmup_iters), 1.0)
            gate = (1.0 - ratio) * torch.ones_like(gate) + ratio * gate
        return gate.unsqueeze(1).detach()

    def _build_boundary_pair_response(self, label):
        if self.doris_edge:
            return self._build_doris_boundary_pair_response(label)
        pair_gate = self._build_boundary_pair_gate(label)
        min_weight = max(float(self.struct_edge_pair_min_weight), 1e-6)
        response = (pair_gate - min_weight) / max(1.0 - min_weight, 1e-6)
        return response.clamp(0.0, 1.0)

    def _build_doris_boundary_pair_response(self, label):
        label = self._as_struct_edge_label(label)
        response = label.new_zeros(label.shape, dtype=torch.float)
        if not self.struct_edge_use_pair_prior or not self.struct_edge_pair_prior_ready.item():
            return response.unsqueeze(1)
        valid = self._valid_struct_edge_label(label)
        count = torch.zeros_like(response)
        prior = torch.nan_to_num(self.struct_edge_pair_prior.to(response),
                                  nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = torch.roll(label, shifts=(dy, dx), dims=(1, 2))
            shifted_valid = torch.roll(valid, shifts=(dy, dx), dims=(1, 2))
            if dy > 0:
                shifted_valid[:, :dy, :] = False
            elif dy < 0:
                shifted_valid[:, dy:, :] = False
            if dx > 0:
                shifted_valid[:, :, :dx] = False
            elif dx < 0:
                shifted_valid[:, :, dx:] = False
            boundary = valid & shifted_valid & label.ne(shifted)
            score = prior[label.clamp(0, self.struct_edge_num_classes - 1),
                          shifted.clamp(0, self.struct_edge_num_classes - 1)]
            if self.struct_edge_cfg.get('pair_response_mode', 'legacy') != 'valid_boundary_only':
                score = score.pow(self.struct_edge_pair_power)
            response = response + torch.where(boundary, score, torch.zeros_like(score))
            count = count + boundary.float()
        response = response / count.clamp_min(1)
        if self.struct_edge_cfg.get('pair_response_mode', 'legacy') == 'valid_boundary_only':
            return response.clamp(0, 1).unsqueeze(1).detach()
        ramp = min(1.0, float(self.local_iter) /
                   max(float(self.struct_edge_pair_warmup_iters), 1.0))
        if self.struct_edge_pair_warmup_iters <= 0:
            ramp = 1.0
        return (response * ramp).clamp(0, 1).unsqueeze(1).detach()

    def doris_edge_responses(self, img, label, target_probs=None):
        """Shared by DORIS training and the independent response exporter."""
        image = self._build_img_edge_response(img)
        probability = (self._build_prob_edge_response(target_probs)
                       if target_probs is not None else None)
        pair = (self._build_boundary_pair_response(label)
                if target_probs is not None else None)
        numerator, denominator = torch.zeros_like(image), torch.zeros_like(image)
        for coefficient, value in ((self.struct_edge_img_weight, image),
                                   (self.struct_edge_prob_weight, probability),
                                   (self.struct_edge_pair_weight, pair)):
            if value is None:
                continue
            if value.shape[-2:] != image.shape[-2:]:
                value = F.interpolate(value, size=image.shape[-2:], mode='bilinear',
                                      align_corners=False)
            available = torch.isfinite(value)
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
            numerator = numerator + coefficient * value * available
            denominator = denominator + coefficient * available
        fused = (numerator / denominator.clamp_min(1e-6)).clamp(0, 1)
        def finite_response(value):
            return (torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
                    if value is not None else None)
        return dict(E_img=finite_response(image), E_prob=finite_response(probability),
                    R_pair=finite_response(pair), R_edge=fused)

    def _doris_target_valid_mask(self, label, valid_pseudo_mask=None):
        # Prototype/structure-only observation must not depend on edge attributes.
        valid = ((label != getattr(self, 'struct_edge_ignore_index', 255)) &
                 (label >= 0) & (label < getattr(
                     self, 'struct_edge_num_classes', self.num_classes))).unsqueeze(1)
        if valid_pseudo_mask is not None:
            if valid_pseudo_mask.dim() == 3:
                valid_pseudo_mask = valid_pseudo_mask.unsqueeze(1)
            valid = valid & valid_pseudo_mask.gt(0)
        if self.psweight_ignore_top > 0:
            valid[:, :, :self.psweight_ignore_top] = False
        if self.psweight_ignore_bottom > 0:
            valid[:, :, -self.psweight_ignore_bottom:] = False
        return valid.detach()

    def mechanism_boundary(self, label):
        """Observation only: reuse the exact boundary target extraction rule."""
        return label_to_edge(label, self.struct_edge_ignore_index,
                             self.struct_edge_radius).bool()

    def _doris_mix_valid_mask(self, mix_mask, source_label, target_valid):
        if mix_mask.dim() == 3:
            mix_mask = mix_mask.unsqueeze(0)
        source_valid = self._valid_struct_edge_label(source_label).reshape_as(mix_mask)
        valid = torch.where(mix_mask.bool(), source_valid, target_valid.bool())
        # Remove the synthetic seam plus the boundary-label and negative-ring
        # receptive fields. This mask also blocks negative-weight propagation.
        seam = label_to_edge(mix_mask[:, 0].long(),
                             edge_radius=self.struct_edge_radius + 3).bool()
        return (valid & ~seam).detach()

    def _doris_log_enabled(self):
        return (self.doris_process_logging and self.doris_log_interval > 0 and
                (self.local_iter + 1) % self.doris_log_interval == 0)

    def _mechanism_sample_enabled(self):
        return (self.mechanism_logging and
                (self.local_iter + 1) % self.mechanism_cfg['sample_interval'] == 0)

    def _log_doris_process(self, confidence, label, pseudo_weight, valid_mask,
                            struct_debug, log_vars, w_t_ps=None, w_t_phy=None,
                            r_phy=None):
        if not self._doris_log_enabled():
            return
        valid = (valid_mask[:, 0] if valid_mask is not None else
                 label.ne(255))
        high = confidence.ge(self.pseudo_threshold)
        log_vars['doris.teacher_spatial_valid_ratio'] = valid.float().mean().item()
        log_vars['doris.teacher_high_conf_ratio_valid'] = (
            high[valid].float().mean().item() if valid.any() else 0.0)
        for cls, name in ((0, 'goji'), (1, 'vegetation')):
            pixels = valid & label.eq(cls)
            log_vars[f'doris.teacher_{name}_prediction_ratio_valid'] = (
                pixels.float().sum() / valid.sum().clamp_min(1)).item()
            log_vars[f'doris.teacher_{name}_high_conf_ratio_given_class'] = (
                high[pixels].float().mean().item() if pixels.any() else 0.0)
            log_vars[f'doris.{name}_final_pseudo_weight_mean_valid'] = (
                pseudo_weight[pixels].mean().item() if pixels.any() else 0.0)
        if self.enable_proto:
            log_vars['doris.target_prototype_valid_classes'] = \
                self.proto_gate.target_valid_proto.float().sum().item()
        if struct_debug is not None:
            response = struct_debug['R_struct']
            struct_mask = self._class_mask(label, self.struct_gate.struct_positive_classes) & valid
            log_vars['doris.R_struct_mean_all'] = response.mean().item()
            log_vars['doris.R_struct_mean_struct_classes_valid'] = (
                response[struct_mask].mean().item() if struct_mask.any() else 0.0)
        def mean_valid(value, mask=valid):
            if value.dim() == 4:
                value = value[:, 0]
            return value[mask].mean().item() if mask.any() else 0.0
        log_vars['doris.teacher.w_conf_mean'] = mean_valid(confidence)
        log_vars['doris.teacher.dacs_high_conf_ratio_all'] = high.float().mean().item()
        log_vars['doris.teacher.spatial_valid_ratio'] = valid.float().mean().item()
        log_vars['doris.teacher.high_conf_ratio_valid'] = (
            high[valid].float().mean().item() if valid.any() else 0.0)
        for cls, name in enumerate(self.class_names):
            pixels = valid & label.eq(cls)
            log_vars['doris.teacher.' + name + '.prediction_ratio_valid'] = (
                pixels.sum().float() / valid.sum().clamp_min(1)).item()
            log_vars['doris.pbcr.proto.' + name + '.w_t_final_mean'] = mean_valid(pseudo_weight, pixels)
        log_vars['doris.mparc.w_t_ps_mean'] = mean_valid(
            w_t_ps if w_t_ps is not None else pseudo_weight)
        log_vars['doris.pbcr.w_t_final_mean'] = mean_valid(pseudo_weight)
        if w_t_phy is not None:
            log_vars['doris.mparc.w_t_phy_mean'] = mean_valid(w_t_phy)
        if r_phy is not None:
            if isinstance(r_phy, dict):
                log_vars['doris.mparc.s_phy_class_agreement_mean'] = mean_valid(
                    r_phy['s_phy_class_agreement'])
            else:
                log_vars['doris.mparc.r_phy_mean'] = mean_valid(r_phy.detach())
        if struct_debug is not None:
            for name, value in (('r_conf', confidence),
                                ('r_loc', struct_debug.get('R_continuity')),
                                ('r_adj', struct_debug.get('R_adj')),
                                ('r_prob', struct_debug.get('R_sem')),
                                ('r_struct', struct_debug['R_struct'])):
                if value is not None:
                    log_vars['doris.mparc.' + name + '_mean'] = mean_valid(value)
            for name, bank in (('adj', self.struct_gate.source_adj_initialized),
                                ('prob', self.struct_gate.semantic_proto_initialized)):
                log_vars['doris.mparc.' + name + '_ready_classes'] = bank.float().sum().item()
            ratio = self.struct_gate.gate_strength(self.local_iter)
            log_vars['doris.schedule.struct_ramp'] = ratio
            log_vars['doris.schedule.struct_feature_strength'] = 0.05 * ratio
        if self.enable_struct_edge:
            log_vars['doris.schedule.edge_ramp'] = self._struct_edge_warmup_ratio()
        if self.enable_proto:
            gate = self.proto_gate
            log_vars['doris.pbcr.tau_proto'] = gate.tau_proto
            log_vars['doris.pbcr.tau_conf_proto'] = gate.tau_conf_proto
            log_vars['doris.schedule.proto_weight_active'] = float(self.local_iter >= gate.weight_start)
            log_vars['doris.schedule.proto_loss_active'] = float(self.local_iter >= gate.loss_start)
            for key, value in gate.last_target_diagnostics.items():
                for cls, name in enumerate(self.class_names):
                    key = key.replace('class_{}.'.format(cls), 'proto.' + name + '.')
                log_vars['doris.pbcr.' + key] = (value.detach().item()
                                               if torch.is_tensor(value) else value)
            for cls, name in enumerate(self.class_names):
                log_vars['doris.pbcr.proto.' + name + '.source_ready'] = float(
                    gate.source_valid_proto[cls])
                log_vars['doris.pbcr.proto.' + name + '.target_ready'] = float(
                    gate.target_valid_proto[cls])

    def _parse_patch_position_from_meta(self, img_meta):
        filename = img_meta.get('ori_filename',
                                img_meta.get('filename', ''))
        stem = os.path.splitext(os.path.basename(filename))[0]
        parts = stem.split('_')
        if len(parts) < 5:
            return None
        try:
            image_id = parts[0]
            x1, y1, x2, y2 = [int(v) for v in parts[-4:]]
        except ValueError:
            return None
        return dict(image_id=image_id, x1=x1, y1=y1, x2=x2, y2=y2)

    def _patch_border_sides(self, patch_meta, img_meta):
        x1, y1 = patch_meta['x1'], patch_meta['y1']
        width = patch_meta['x2'] - patch_meta['x1']
        height = patch_meta['y2'] - patch_meta['y1']
        stride = max(int(self.struct_edge_spatial_stride), 1)
        left = x1 == 0
        top = y1 == 0
        right = (x1 > 0 and width > 0 and x1 % stride != 0)
        bottom = (y1 > 0 and height > 0 and y1 % stride != 0)

        if img_meta.get('flip', False):
            direction = img_meta.get('flip_direction', None)
            if direction == 'horizontal':
                left, right = right, left
            elif direction == 'vertical':
                top, bottom = bottom, top
        return left, right, top, bottom

    def _build_spatial_layout_gate(self, img, img_metas):
        gate = img.new_ones((img.shape[0], 1, img.shape[2], img.shape[3]))
        if not self.struct_edge_use_spatial_prior or img_metas is None:
            return gate
        border_width = int(self.struct_edge_spatial_border_width)
        for i, img_meta in enumerate(img_metas):
            patch_meta = self._parse_patch_position_from_meta(img_meta)
            if patch_meta is None:
                continue
            left, right, top, bottom = self._patch_border_sides(
                patch_meta, img_meta)
            num_sides = int(left) + int(right) + int(top) + int(bottom)
            if num_sides >= 2:
                patch_weight = self.struct_edge_spatial_corner_weight
            elif num_sides == 1:
                patch_weight = self.struct_edge_spatial_edge_weight
            else:
                patch_weight = self.struct_edge_spatial_middle_weight
            gate[i].mul_(patch_weight)
            if border_width <= 0 or num_sides == 0:
                continue
            border_weight = min(self.struct_edge_spatial_border_min_weight,
                                patch_weight)
            if left:
                gate[i, :, :, :border_width] *= border_weight / patch_weight
            if right:
                gate[i, :, :, -border_width:] *= border_weight / patch_weight
            if top:
                gate[i, :, :border_width, :] *= border_weight / patch_weight
            if bottom:
                gate[i, :, -border_width:, :] *= border_weight / patch_weight
        return gate.detach()

    def _patch_memory_key(self, patch_meta):
        return (patch_meta['image_id'], patch_meta['x1'], patch_meta['y1'],
                patch_meta['x2'], patch_meta['y2'])

    def _patch_center(self, patch_meta):
        return ((patch_meta['x1'] + patch_meta['x2']) * 0.5,
                (patch_meta['y1'] + patch_meta['y2']) * 0.5)

    def _label_class_histogram(self, label):
        label = self._as_struct_edge_label(label)
        if label.dim() == 2:
            label = label.unsqueeze(0)
        hists = []
        for item in label:
            valid = self._valid_struct_edge_label(item)
            if valid.any():
                hist = torch.bincount(
                    item[valid].reshape(-1),
                    minlength=self.struct_edge_num_classes).float()
                hist = hist / hist.sum().clamp_min(1.0)
            else:
                hist = item.new_ones(
                    self.struct_edge_num_classes,
                    dtype=torch.float) / self.struct_edge_num_classes
            hists.append(hist)
        return torch.stack(hists, dim=0)

    def _find_neighbor_patch_hists(self, patch_meta):
        stride = max(float(self.struct_edge_spatial_stride), 1.0)
        max_main_dist = stride * 1.25
        max_cross_dist = stride * 0.35
        cx, cy = self._patch_center(patch_meta)
        neighbors = []
        current_key = self._patch_memory_key(patch_meta)
        for key, memory in self.struct_edge_neighbor_memory.items():
            if key == current_key or key[0] != patch_meta['image_id']:
                continue
            other = dict(
                image_id=key[0], x1=key[1], y1=key[2], x2=key[3], y2=key[4])
            nx, ny = self._patch_center(other)
            dx, dy = abs(cx - nx), abs(cy - ny)
            is_horizontal_neighbor = dx <= max_main_dist and dy <= max_cross_dist
            is_vertical_neighbor = dy <= max_main_dist and dx <= max_cross_dist
            if is_horizontal_neighbor or is_vertical_neighbor:
                neighbors.append(memory['hist'])
        return neighbors

    def _build_neighbor_patch_gate(self, label, img_metas):
        label = self._as_struct_edge_label(label)
        gate = label.new_ones(label.shape, dtype=torch.float)
        if not self.struct_edge_use_neighbor_prior or img_metas is None:
            return gate.unsqueeze(1)
        hists = self._label_class_histogram(label)
        for i, img_meta in enumerate(img_metas):
            patch_meta = self._parse_patch_position_from_meta(img_meta)
            if patch_meta is None:
                continue
            neighbor_hists = self._find_neighbor_patch_hists(patch_meta)
            if not neighbor_hists:
                continue
            neighbor_hist = torch.stack(neighbor_hists, dim=0).mean(dim=0)
            neighbor_hist = neighbor_hist.to(device=label.device,
                                             dtype=hists.dtype)
            sim = F.cosine_similarity(
                hists[i].unsqueeze(0),
                neighbor_hist.unsqueeze(0),
                dim=1,
                eps=1e-6).squeeze(0).clamp(0.0, 1.0)
            sim = sim.pow(self.struct_edge_neighbor_power)
            patch_gate = self.struct_edge_neighbor_min_weight + (
                1.0 - self.struct_edge_neighbor_min_weight) * sim
            gate[i].mul_(patch_gate)
        if self.local_iter < self.struct_edge_neighbor_warmup_iters:
            ratio = float(self.local_iter) / max(
                float(self.struct_edge_neighbor_warmup_iters), 1.0)
            gate = (1.0 - ratio) * torch.ones_like(gate) + ratio * gate
        return gate.unsqueeze(1).detach()

    @torch.no_grad()
    def _update_neighbor_patch_memory(self, label, img_metas):
        if not self.struct_edge_use_neighbor_prior or img_metas is None:
            return
        hists = self._label_class_histogram(label).detach().cpu()
        for i, img_meta in enumerate(img_metas):
            patch_meta = self._parse_patch_position_from_meta(img_meta)
            if patch_meta is None:
                continue
            key = self._patch_memory_key(patch_meta)
            hist = hists[i]
            if key in self.struct_edge_neighbor_memory:
                old_hist = self.struct_edge_neighbor_memory[key]['hist']
                hist = (self.struct_edge_neighbor_momentum * old_hist +
                        (1.0 - self.struct_edge_neighbor_momentum) * hist)
            self.struct_edge_neighbor_memory[key] = dict(hist=hist)
        while len(self.struct_edge_neighbor_memory) > \
                self.struct_edge_neighbor_max_items:
            self.struct_edge_neighbor_memory.pop(
                next(iter(self.struct_edge_neighbor_memory)))

    def _edge_ratio(self, label):
        edge = label_to_edge(
            label, edge_radius=self.struct_edge_radius).float()
        return edge.mean().detach().item()

    def _build_struct_edge_weight(self,
                                  img,
                                  label,
                                  loss_weight,
                                  log_vars,
                                  prefix,
                                  base_weight=None,
                                  proto_debug=None,
                                  vi=None,
                                  img_metas=None,
                                  target_probs=None,
                                  is_source=False,
                                  is_target=False,
                                  target_confidence=None,
                                  boundary_valid_mask=None):
        if not self.enable_struct_edge or not self.struct_edge_use_edge_loss:
            return None
        if is_source:
            self._update_boundary_pair_prior(label)
        warmup = self._struct_edge_warmup_ratio()
        if self.doris_edge:
            if is_target and target_probs is None:
                raise ValueError('DORIS target boundary weighting requires Teacher probabilities')
            responses = self.doris_edge_responses(
                img, label, target_probs if is_target else None)
            response = responses['R_edge']
            if is_target and self._mechanism_sample_enabled():
                self.mechanism_edge_response = response.detach()
                self.mechanism_edge_components = {
                    key: value.detach() for key, value in responses.items() if value is not None}
            reliability = torch.ones_like(response)
            if is_target:
                confidence = (target_confidence if target_confidence is not None
                              else target_probs.max(dim=1)[0])
                if confidence.dim() == 3:
                    confidence = confidence.unsqueeze(1)
                confidence = torch.nan_to_num(confidence.detach(), nan=0.0,
                                               posinf=0.0, neginf=0.0)
                reliability = ((confidence - self.struct_edge_cfg.get('confidence_floor', 0.60)) /
                               self.struct_edge_cfg.get('confidence_range', 0.35)).clamp(0, 1)
            if boundary_valid_mask is not None:
                reliability = reliability * boundary_valid_mask.detach().to(reliability)
            if self._doris_log_enabled():
                boundary = label_to_edge(label, self.struct_edge_ignore_index,
                                          self.struct_edge_radius).bool()
                boundary = boundary & self._valid_struct_edge_label(
                    self._as_struct_edge_label(label)).unsqueeze(1)
                if boundary_valid_mask is not None:
                    boundary = boundary & boundary_valid_mask.bool()
                log_vars[f'doris.R_edge_{prefix}_mean_all'] = response.mean().item()
                log_vars[f'doris.R_edge_{prefix}_mean_valid_boundary'] = (
                    response[boundary].mean().item() if boundary.any() else 0.0)
                log_vars[f'doris.valid_boundary_ratio_{prefix}'] = boundary.float().mean().item()
                for name, key in (('e_prob', 'E_prob'), ('e_img', 'E_img'),
                                   ('r_pair', 'R_pair'), ('r_edge', 'R_edge')):
                    value = responses.get(key)
                    if value is not None:
                        log_vars['doris.pbcr.' + prefix + '.' + name + '_mean_boundary'] = (
                            value[boundary].mean().item() if boundary.any() else 0.0)
                log_vars['doris.pbcr.' + prefix + '.reliability_mean_boundary'] = (
                    reliability[boundary].mean().item() if boundary.any() else 0.0)
            return (response * reliability * (loss_weight * warmup)).detach()
        if self.struct_edge_mode == 'paper_pbcr':
            img_edge = self._build_img_edge_response(img)
            prob_edge = (self._build_prob_edge_response(target_probs)
                         if target_probs is not None else
                         img.new_zeros(img_edge.shape))
            pair_response = (self._build_boundary_pair_response(label)
                             if is_target else img.new_zeros(img_edge.shape))
            if prob_edge.shape[-2:] != img_edge.shape[-2:]:
                prob_edge = F.interpolate(
                    prob_edge,
                    size=img_edge.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            if pair_response.shape[-2:] != img_edge.shape[-2:]:
                pair_response = F.interpolate(
                    pair_response,
                    size=img_edge.shape[-2:],
                    mode='nearest')
            gate = (
                self.struct_edge_prob_weight * prob_edge +
                self.struct_edge_img_weight * img_edge +
                self.struct_edge_pair_weight * pair_response).clamp(0.0, 1.0)
            if self.log_debug_stats:
                if is_target:
                    log_vars[f'prob_edge_{prefix}_mean'] = \
                        prob_edge.detach().mean().item()
                    log_vars[f'img_edge_{prefix}_mean'] = \
                        img_edge.detach().mean().item()
                    log_vars[f'boundary_pair_response_{prefix}_mean'] = \
                        pair_response.detach().mean().item()
                    log_vars[f'boundary_pair_prior_ready_{prefix}'] = \
                        float(self.struct_edge_pair_prior_ready.item())
                else:
                    log_vars[f'img_edge_{prefix}_mean'] = \
                        img_edge.detach().mean().item()
            edge_weight = gate * (loss_weight * warmup)
            if self.log_debug_stats:
                log_vars[f'struct_edge_w_{prefix}_mean'] = \
                    edge_weight.detach().mean().item()
                log_vars[f'edge_weight_{prefix}_mean'] = \
                    edge_weight.detach().mean().item()
                log_vars[f'fused_gate_{prefix}_mean'] = gate.detach().mean().item()
                log_vars[f'ss_gate_{prefix}_mean'] = gate.detach().mean().item()
                log_vars[f'struct_edge_ratio_{prefix}'] = self._edge_ratio(label)
            return (edge_weight.detach()
                    if self.struct_edge_detach_gate else edge_weight)
        struct_gate = (self._build_img_struct_gate(img)
                       if self.struct_edge_use_struct_gate else None)
        spec_gate = None
        if self.struct_edge_use_spec_gate:
            vi = self._get_vi_tensor(img, explicit_vi=vi)
            spec_gate = self._build_spec_gate(vi)
        gate = self._fuse_struct_spec_gate(
            struct_gate, spec_gate, ref_tensor=img)
        spatial_gate = self._build_spatial_layout_gate(img, img_metas)
        gate = gate * spatial_gate
        if self.log_debug_stats:
            log_vars[f'spatial_layout_gate_{prefix}_mean'] = \
                spatial_gate.detach().mean().item()
        if is_target:
            proto_gate = None
            if proto_debug is not None:
                proto_gate = proto_debug.get('proto_weight')
            gate = self._apply_target_reliability_gate(
                gate, pseudo_weight=base_weight, proto_gate=proto_gate)
            pair_gate = self._build_boundary_pair_gate(label)
            gate = gate * pair_gate
            if self.log_debug_stats:
                log_vars[f'boundary_pair_gate_{prefix}_mean'] = \
                    pair_gate.detach().mean().item()
                log_vars[f'boundary_pair_prior_ready_{prefix}'] = \
                    float(self.struct_edge_pair_prior_ready.item())
            neighbor_gate = self._build_neighbor_patch_gate(label, img_metas)
            gate = gate * neighbor_gate
            if self.log_debug_stats:
                log_vars[f'neighbor_patch_gate_{prefix}_mean'] = \
                    neighbor_gate.detach().mean().item()
                log_vars[f'neighbor_patch_memory_size_{prefix}'] = \
                    float(len(self.struct_edge_neighbor_memory))
        edge_weight = gate * (loss_weight * warmup)
        if self.log_debug_stats:
            log_vars[f'struct_edge_w_{prefix}_mean'] = \
                edge_weight.detach().mean().item()
            log_vars[f'edge_weight_{prefix}_mean'] = \
                edge_weight.detach().mean().item()
            log_vars['edge_mode'] = self.local_iter * 0.0
            if struct_gate is not None:
                log_vars[f'img_struct_gate_{prefix}_mean'] = \
                    struct_gate.detach().mean().item()
                log_vars[f'struct_gate_{prefix}_mean'] = \
                    struct_gate.detach().mean().item()
            if spec_gate is not None:
                log_vars[f'spec_gate_{prefix}_mean'] = \
                    spec_gate.detach().mean().item()
                log_vars[f'spec_gate_{prefix}_max'] = \
                    spec_gate.detach().amax().item()
            log_vars[f'fused_gate_{prefix}_mean'] = gate.detach().mean().item()
            log_vars[f'ss_gate_{prefix}_mean'] = gate.detach().mean().item()
            log_vars[f'struct_edge_ratio_{prefix}'] = self._edge_ratio(label)
        if is_target:
            self._update_neighbor_patch_memory(label, img_metas)
        return edge_weight.detach() if self.struct_edge_detach_gate else edge_weight

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None,
                      source_vi_img=None,
                      target_vi_img=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)
        self.update_debug_state()
        seg_debug = {}

        src_means, src_stds = get_mean_std(img_metas, dev)
        trg_means, trg_stds = get_mean_std(target_img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': src_means[0].unsqueeze(0),
            'std': src_stds[0].unsqueeze(0)
        }

        # Train on source images
        src_struct_edge_weight = self._build_struct_edge_weight(
            img,
            gt_semantic_seg,
            self.struct_edge_loss_weight_src,
            log_vars,
            'src',
            vi=source_vi_img,
            img_metas=img_metas,
            is_source=True) if self.enable_struct_edge else None
        clean_losses = self.get_model().forward_train(
            img,
            img_metas,
            gt_semantic_seg,
            struct_edge_weight=src_struct_edge_weight,
            return_feat=True)
        src_feat = clean_losses.pop('features')
        if self.save_class_mix_debug:
            seg_debug['Source'] = self.get_model().decode_head.debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.enable_phy:
            if source_vi_img is None or target_vi_img is None:
                raise ValueError('phy_cfg.enable=True requires source_vi_img '
                                 'and target_vi_img from the dataloader.')
            r_phy_src, loss_phy_src = self._compute_phy_src_loss(
                source_vi_img, gt_semantic_seg, log_vars)
            loss_phy_src.backward()
            del r_phy_src, loss_phy_src
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        if self.enable_proto and self.proto_cfg.get('source_update', True):
            proto_src_log = self.proto_gate.update_source(
                src_feat, gt_semantic_seg, iter=self.local_iter)
            self._log_proto_update('proto_src', proto_src_log, log_vars)
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().generate_pseudo_label(
            target_img, target_img_metas)
        if self.save_class_mix_debug:
            seg_debug['Target'] = self.get_ema_model().decode_head.debug_output

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        del ema_logits
        teacher_target_confidence_map, pseudo_label = torch.max(
            ema_softmax, dim=1)
        teacher_raw_pseudo_label = pseudo_label
        if valid_pseudo_mask is not None:
            valid_pseudo_mask = valid_pseudo_mask.float().clamp(0, 1)
        if self.log_debug_stats or self.paper_process_logging:
            log_vars['teacher_target_confidence_mean'] = \
                teacher_target_confidence_map.mean().item()
        if self.log_debug_stats and not self.doris_edge:
            retain_mask = teacher_target_confidence_map.ge(self.pseudo_threshold)
            if valid_pseudo_mask is not None:
                teacher_pseudo_label_retain_ratio = valid_pseudo_mask.mean().item()
            else:
                teacher_pseudo_label_retain_ratio = retain_mask.float().mean().item()
            log_vars['teacher_pseudo_label_retain_ratio'] = \
                teacher_pseudo_label_retain_ratio
        self._maybe_save_teacher_pseudo_debug(teacher_raw_pseudo_label,
                                              teacher_target_confidence_map)
        pseudo_prob = teacher_target_confidence_map
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)
        del pseudo_prob, ps_large_p, ps_size

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        r_phy_tgt, phy_weight = None, None
        original_pseudo_weight = pseudo_weight.detach().clone()
        phy_pseudo_weight = pseudo_weight
        struct_pseudo_weight = pseudo_weight
        if self.enable_phy:
            phy_input_weight = (original_pseudo_weight.clone()
                                if self.enable_joint_gate else pseudo_weight)
            r_phy_tgt, phy_pseudo_weight, phy_weight = self._apply_phy_gate(
                target_vi_img, pseudo_label, phy_input_weight, log_vars,
                valid_mask=self._doris_target_valid_mask(pseudo_label, valid_pseudo_mask))
            if not self.enable_joint_gate:
                pseudo_weight = phy_pseudo_weight
        struct_debug = None
        orig_struct_pseudo_weight = pseudo_weight.detach().clone()
        if self.enable_struct:
            source_probs = None
            if self.struct_cfg.get('use_semantic_proto', True):
                with torch.no_grad():
                    source_logits = self.get_model().generate_pseudo_label(
                        img, img_metas)
                    source_probs = torch.softmax(source_logits.detach(), dim=1)
                    del source_logits
            struct_input_weight = (original_pseudo_weight.clone()
                                   if self.enable_joint_gate else pseudo_weight)
            struct_pseudo_weight, struct_debug = self._apply_struct_gate(
                ema_softmax, pseudo_label, struct_input_weight,
                gt_semantic_seg, log_vars, source_probs=source_probs,
                target_confidence=(teacher_target_confidence_map
                    if self.struct_gate.fusion_mode == 'doris_confidence_guided' else None))
            if not self.enable_joint_gate:
                pseudo_weight = struct_pseudo_weight
        struct_prior_response = None
        if struct_debug is not None and self.struct_cfg.get(
                'use_feature_interaction', False):
            feat_warmup = self.struct_cfg.get(
                'feature_interaction_warmup_iters',
                self.struct_cfg.get('struct_warmup_iters', 0))
            feat_rampup = self.struct_cfg.get(
                'feature_interaction_rampup_iters',
                self.struct_cfg.get('struct_rampup_iters', 1))
            if self.local_iter >= feat_warmup or self.struct_gate.struct_gate_schedule == 'early_nonzero_linear':
                feat_ratio = self.struct_gate.feature_strength(self.local_iter, feat_warmup, feat_rampup)
                struct_prior_response = \
                    struct_debug['R_struct'].unsqueeze(1) * feat_ratio
                if self.log_debug_stats:
                    log_vars['struct_feature_ratio'] = feat_ratio
        joint_debug = None
        if self.enable_joint_gate:
            pseudo_weight, joint_debug = self._fuse_phy_struct_pseudo_weight(
                original_pseudo_weight,
                phy_pseudo_weight if self.enable_phy else original_pseudo_weight,
                struct_pseudo_weight
                if self.enable_struct else original_pseudo_weight,
                pseudo_label, log_vars)
        edge_conf_pseudo_weight = pseudo_weight.detach().clone()
        if self.paper_process_logging:
            # MPARC stage: after physiology/structure, before prototype weighting.
            log_vars['mparc_pseudo_weight_mean'] = pseudo_weight.detach().mean().item()
        target_feat = None
        proto_debug = None
        w_t_ps = edge_conf_pseudo_weight
        if self.enable_proto:
            self.proto_gate.collect_diagnostics = (
                self._doris_log_enabled() or self._mechanism_sample_enabled())
            if self.proto_gate.collect_diagnostics:
                self.proto_gate.diagnostic_spatial_mask = self._doris_target_valid_mask(
                    pseudo_label, valid_pseudo_mask)
            target_feat = self.get_model().extract_feat(target_img)
            proto_tgt_log = self.proto_gate.update_target(
                target_feat, pseudo_label, teacher_target_confidence_map,
                w_t_ps, iter=self.local_iter)
            self.proto_gate.diagnostic_spatial_mask = None
            self._log_proto_update('proto_tgt', proto_tgt_log, log_vars)
            with torch.no_grad():
                proto_weight, proto_debug = self.proto_gate.compute_weight(
                    target_feat,
                    pseudo_label,
                    out_size=pseudo_label.shape[-2:],
                    iter=self.local_iter)
            self._log_proto_debug(proto_debug, log_vars)
            pseudo_weight = pseudo_weight * proto_weight.detach()
            del proto_weight
        if self.log_debug_stats:
            log_vars['final_pseudo_weight_mean'] = \
                pseudo_weight.detach().mean().item()
        final_gate_pseudo_weight = pseudo_weight.detach().clone()
        tgt_boundary_valid = (self._doris_target_valid_mask(pseudo_label, valid_pseudo_mask)
                              if self.doris_edge else None)
        self._log_doris_process(teacher_target_confidence_map, pseudo_label,
                                pseudo_weight, (tgt_boundary_valid if self.doris_edge else
                                                valid_pseudo_mask), struct_debug, log_vars,
                                w_t_ps=w_t_ps, w_t_phy=phy_pseudo_weight, r_phy=r_phy_tgt)
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)
        tgt_struct_edge_weight = self._build_struct_edge_weight(
            target_img,
            pseudo_label,
            self.struct_edge_loss_weight_tgt,
            log_vars,
            'tgt',
            base_weight=edge_conf_pseudo_weight,
            proto_debug=proto_debug,
            vi=target_vi_img,
            img_metas=target_img_metas,
            target_probs=ema_softmax.detach(),
            is_target=True,
            target_confidence=teacher_target_confidence_map,
            boundary_valid_mask=tgt_boundary_valid) if self.enable_struct_edge else None

        if self._mechanism_sample_enabled():
            self.mechanism_records.append(training_record(
                self, teacher_target_confidence_map, pseudo_label, w_t_ps,
                pseudo_weight, self._doris_target_valid_mask(pseudo_label, valid_pseudo_mask),
                struct_debug, getattr(self, 'mechanism_edge_response', None),
                tgt_struct_edge_weight, w_base=original_pseudo_weight,
                w_t_phy=phy_pseudo_weight, r_phy=r_phy_tgt,
                edge_components=getattr(self, 'mechanism_edge_components', None),
                log_vars=log_vars))
            if self.enable_struct_edge:
                del self.mechanism_edge_response, self.mechanism_edge_components

        mixed_img, mixed_lbl, mix_masks = None, None, None
        if self.enable_mixing:
            # Align target images to source normalization before ClassMix.
            target_img_mix = convert_to_norm(target_img, trg_means, trg_stds,
                                             src_means, src_stds)

            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mixed_struct_edge_weight = (
                [None] * batch_size if self.enable_struct_edge else None)
            mixed_boundary_valid = [] if self.doris_edge else None
            mixed_struct_prior_response = (
                [None] * batch_size
                if struct_prior_response is not None else None)
            mix_masks = get_class_masks(gt_semantic_seg)

            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img_mix[i])),
                    target=torch.stack((gt_semantic_seg[i][0],
                                        pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
                if self.enable_struct_edge:
                    _, mixed_struct_edge_weight[i] = strong_transform(
                        strong_parameters,
                        target=torch.stack((
                            src_struct_edge_weight[i, 0],
                            tgt_struct_edge_weight[i, 0])))
                    if self.doris_edge:
                        mixed_boundary_valid.append(self._doris_mix_valid_mask(
                            mix_masks[i], gt_semantic_seg[i], tgt_boundary_valid[i:i + 1]))
                if struct_prior_response is not None:
                    source_prior = torch.ones_like(
                        struct_prior_response[i, 0])
                    _, mixed_struct_prior_response[i] = strong_transform(
                        strong_parameters,
                        target=torch.stack((
                            source_prior, struct_prior_response[i, 0])))
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)
            boundary_kwargs = (dict(struct_edge_valid_mask=torch.cat(mixed_boundary_valid))
                               if self.doris_edge else {})
            if struct_prior_response is not None:
                mixed_struct_prior_response = torch.cat(
                    mixed_struct_prior_response)
                if mixed_struct_prior_response.dim() == 3:
                    mixed_struct_prior_response = \
                        mixed_struct_prior_response.unsqueeze(1)
            else:
                mixed_struct_prior_response = None
            if self.enable_struct_edge:
                mixed_struct_edge_weight = torch.cat(
                    mixed_struct_edge_weight)
                if self.log_debug_stats or self.paper_process_logging:
                    log_vars['struct_edge_w_mix_mean'] = \
                        mixed_struct_edge_weight.detach().mean().item()
                if self.log_debug_stats:
                    log_vars['struct_edge_ratio_mix'] = self._edge_ratio(mixed_lbl)
            else:
                mixed_struct_edge_weight = None

            # Train on mixed images
            mix_losses = self.get_model().forward_train(
                mixed_img,
                img_metas,
                mixed_lbl,
                pseudo_weight,
                struct_edge_weight=mixed_struct_edge_weight,
                struct_prior_response=mixed_struct_prior_response,
                return_feat=False,
                **boundary_kwargs)
            if self.save_class_mix_debug:
                seg_debug['Mix'] = self.get_model().decode_head.debug_output
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        else:
            boundary_kwargs = (dict(struct_edge_valid_mask=tgt_boundary_valid)
                               if self.doris_edge else {})
            tgt_losses = self.get_model().forward_train(
                target_img,
                target_img_metas,
                pseudo_label.unsqueeze(1),
                pseudo_weight,
                struct_edge_weight=tgt_struct_edge_weight,
                struct_prior_response=struct_prior_response,
                return_feat=False,
                **boundary_kwargs)
            if self.save_class_mix_debug:
                seg_debug['TargetSelfTrain'] = \
                    self.get_model().decode_head.debug_output
            tgt_losses = add_prefix(tgt_losses, 'tgt')
            mix_loss, mix_log_vars = self._parse_losses(tgt_losses)
        del gt_pixel_weight
        log_vars.update(mix_log_vars)
        mix_loss.backward()
        if self.enable_mic and self.mask_mode.startswith('separate'):
            masked_losses = self.mic(
                self.get_model(), img, img_metas, gt_semantic_seg,
                target_img, target_img_metas, valid_pseudo_mask,
                pseudo_label, pseudo_weight)
            if self.save_class_mix_debug and self.mic.debug_output:
                seg_debug.update(self.mic.debug_output)
            mic_losses = {}
            for key, value in masked_losses.items():
                if key == 'decode.loss_seg':
                    mic_losses['loss_mic'] = value
                else:
                    mic_losses[f'mic.{key}'] = value
            mic_loss, mic_log_vars = self._parse_losses(mic_losses)
            log_vars.update(mic_log_vars)
            mic_loss.backward()
        if self.enable_proto:
            loss_proto_tgt, proto_loss_log = self.proto_gate.loss_target(
                target_feat, pseudo_label, final_gate_pseudo_weight,
                iter=self.local_iter)
            loss_proto_tgt = self.proto_cfg.get(
                'lambda_proto_tgt', 0.005) * loss_proto_tgt
            log_vars['loss_proto_tgt'] = loss_proto_tgt.detach().item()
            loss_proto_tgt.backward()
            del loss_proto_tgt, proto_loss_log, target_feat
        if self.enable_phy and self.lambda_phy_tgt > 0:
            loss_phy_tgt, phy_diag = target_phy_loss(
                r_phy_tgt, pseudo_label, teacher_target_confidence_map,
                valid_mask=self._doris_target_valid_mask(pseudo_label, valid_pseudo_mask),
                mode=self.phy_target_supervision_mode, threshold=self.tau_conf_phy,
                positive_classes=self.phy_positive_classes, num_classes=self.num_classes,
                pos_weight=self.phy_pos_weight, neg_weight=self.phy_neg_weight)
            loss_phy_tgt = self.lambda_phy_tgt * loss_phy_tgt
            log_vars['loss_phy_tgt'] = loss_phy_tgt.item()
            if self.phy_mode == 'hierarchical_gv':
                log_vars['loss_phy_tgt_veg_weighted'] = log_vars['loss_phy_tgt']
                log_vars['loss_phy_tgt_gv_weighted'] = 0.0
            if self._mechanism_sample_enabled() and (
                    self.mechanism_cfg['output_mode'] != 'branches' or self.mechanism_cfg['detailed_output']):
                record = self.mechanism_records[-1]
                record['phy_losses']['loss_phy_tgt'] = log_vars['loss_phy_tgt']
                for name, value in phy_diag.items():
                    record['phy_losses'][name] = value.item()
            loss_phy_tgt.backward()

        if self.enable_phy:
            self._maybe_save_phy_debug(
                img, img_metas, source_vi_img, gt_semantic_seg, target_vi_img,
                pseudo_label, teacher_target_confidence_map, r_phy_tgt,
                pseudo_weight, target_img, target_img_metas)

        if self.enable_struct:
            self._maybe_save_struct_debug(
                target_img, target_img_metas, gt_semantic_seg, pseudo_label,
                teacher_target_confidence_map, orig_struct_pseudo_weight,
                struct_debug['final_pseudo_weight'], struct_debug)

        if self.enable_joint_gate:
            self._maybe_save_joint_debug(
                target_img, target_img_metas, target_vi_img, pseudo_label,
                teacher_target_confidence_map, r_phy_tgt, struct_debug,
                original_pseudo_weight,
                phy_pseudo_weight if self.enable_phy else original_pseudo_weight,
                struct_pseudo_weight
                if self.enable_struct else original_pseudo_weight,
                final_gate_pseudo_weight)

        if self.enable_proto:
            self._maybe_save_proto_debug(proto_debug, final_gate_pseudo_weight)

        if self.save_class_mix_debug and self.debug_img_interval > 0 and \
                self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = get_vis_image(img, img_metas, dev)
            vis_trg_img = get_vis_image(target_img, target_img_metas, dev)
            if self.enable_mixing:
                vis_mixed_img = get_vis_image(mixed_img, img_metas, dev)
                for j in range(batch_size):
                    rows, cols = 2, 5
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    subplotimg(axs[0][0], vis_img[j], 'Source Image')
                    subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                    subplotimg(
                        axs[0][1],
                        gt_semantic_seg[j],
                        'Source Seg GT',
                        cmap='cityscapes')
                    subplotimg(
                        axs[1][1],
                        pseudo_label[j],
                        'Target Seg (Pseudo) GT',
                        cmap='cityscapes')
                    subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                    subplotimg(
                        axs[1][2], mix_masks[j][0], 'Domain Mask',
                        cmap='gray')
                    subplotimg(
                        axs[1][3], mixed_lbl[j], 'Seg Targ',
                        cmap='cityscapes')
                    subplotimg(
                        axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0,
                        vmax=1)
                    if self.debug_fdist_mask is not None:
                        subplotimg(
                            axs[0][4],
                            self.debug_fdist_mask[j][0],
                            'FDist Mask',
                            cmap='gray')
                    if self.debug_gt_rescale is not None:
                        subplotimg(
                            axs[1][4],
                            self.debug_gt_rescale[j],
                            'Scaled GT',
                            cmap='cityscapes')
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_{j}.png'))
                    plt.close()

            if seg_debug['Source'] is not None and seg_debug:
                for j in range(batch_size):
                    rows = len(seg_debug)
                    cols = max(len(outs) for outs in seg_debug.values())
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        squeeze=False,
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            if out.ndim == 4 and out.shape[1] >= 3:
                                target_debug = n1 in [
                                    'Target', 'TargetSelfTrain', 'Masked'
                                ]
                                means = trg_means if target_debug else src_means
                                stds = trg_stds if target_debug else src_stds
                                vis = torch.clamp(
                                    denorm(out[:, :3], means[:, :3],
                                           stds[:, :3]), 0, 1)
                                subplotimg(axs[k1][k2], vis[j], f'{n1} {n2}')
                            else:
                                if out.ndim == 3:
                                    args = dict(cmap='cityscapes')
                                else:
                                    args = dict(cmap='gray', vmin=0, vmax=1)
                                subplotimg(axs[k1][k2], out[j], f'{n1} {n2}',
                                           **args)
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_{j}_s.png'))
                    plt.close()
        self.local_iter += 1

        return log_vars
