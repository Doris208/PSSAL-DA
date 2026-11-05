# Adapted from BIT-DA/SePiCo.

import math
import random
from copy import deepcopy

import numpy as np
import torch
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.losses.contrastive_loss import contrast_preparations
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (convert_to_norm,
                                                get_class_masks, get_mean_std,
                                                strong_transform)
from mmseg.models.utils.proto_estimator import ProtoEstimator


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        return max(p.abs().max() for p in grads)
    return torch.norm(
        torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)


@UDA.register_module()
class SePiCo(UDADecorator):

    def __init__(self, **cfg):
        super(SePiCo, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg.get('debug_img_interval', 1000)
        self.print_grad_magnitude = cfg.get('print_grad_magnitude', False)
        self.enable_self_training = cfg['enable_self_training']
        self.enable_strong_aug = cfg['enable_strong_aug']
        self.push_off_self_training = cfg.get('push_off_self_training', False)
        self.pseudo_random_crop = cfg.get('pseudo_random_crop', False)
        if self.pseudo_random_crop:
            raise NotImplementedError(
                'pseudo_random_crop is intentionally disabled for Goji SePiCo.')
        assert self.mix == 'class'

        aux_head = cfg['model']['auxiliary_head']
        self.proj_dim = aux_head['channels']
        self.contrast_mode = aux_head.get('input_transform', None)
        self.calc_layers = aux_head.get('in_index', -1)
        self.num_classes = cfg['model']['decode_head']['num_classes']
        loss_cfg = aux_head['loss_decode']
        self.enable_avg_pool = loss_cfg['use_avg_pool']
        self.scale_min_ratio = loss_cfg['scale_min_ratio']
        self.start_distribution_iter = cfg['start_distribution_iter']
        self.memory_length = cfg.get('memory_length', 0)
        self.ignore_index = 255

        if self.contrast_mode == 'multiple_select':
            self.feat_distributions = {}
            for idx in range(len(self.calc_layers)):
                self.feat_distributions[idx] = ProtoEstimator(
                    dim=self.proj_dim,
                    class_num=self.num_classes,
                    memory_length=self.memory_length)
        else:
            self.feat_distributions = ProtoEstimator(
                dim=self.proj_dim,
                class_num=self.num_classes,
                memory_length=self.memory_length)

        self.ema_model = build_segmentor(deepcopy(cfg['model']))

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:
                ema_param.data = param.data.clone()
            else:
                ema_param.data[:] = param.data[:].clone()

    def _update_ema(self, iteration):
        alpha_teacher = min(1 - 1 / (iteration + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()
        log_vars.pop('loss', None)
        return dict(log_vars=log_vars, num_samples=len(data_batch['img_metas']))

    def _update_distribution(self, feat, gt_semantic_seg):
        mean, covariance, bank = {}, {}, {}
        if self.contrast_mode == 'multiple_select':
            for idx in range(len(self.calc_layers)):
                prep_feat, mask = contrast_preparations(
                    feat[idx], gt_semantic_seg, self.enable_avg_pool,
                    self.scale_min_ratio, self.num_classes, self.ignore_index)
                self.feat_distributions[idx].update_proto(
                    prep_feat.detach(), mask)
                mean[idx] = self.feat_distributions[idx].Ave
                covariance[idx] = self.feat_distributions[idx].CoVariance
                bank[idx] = self.feat_distributions[idx].MemoryBank
        else:
            prep_feat, mask = contrast_preparations(
                feat, gt_semantic_seg, self.enable_avg_pool,
                self.scale_min_ratio, self.num_classes, self.ignore_index)
            self.feat_distributions.update_proto(prep_feat.detach(), mask)
            mean = self.feat_distributions.Ave
            covariance = self.feat_distributions.CoVariance
            bank = self.feat_distributions.MemoryBank
        return mean, covariance, bank

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      target_gt_semantic_seg=None,
                      **kwargs):
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        if self.local_iter == 0:
            self._init_ema_weights()
        elif self.local_iter > 0:
            self._update_ema(self.local_iter)

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

        weak_img = img.clone()
        weak_target_img = target_img.clone()

        for module in self.get_ema_model().modules():
            if isinstance(module, _DropoutNd):
                module.training = False

        ema_target_logits = self.get_ema_model().encode_decode(
            weak_target_img, target_img_metas)
        ema_target_softmax = torch.softmax(ema_target_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_target_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        pseudo_weight = torch.sum(ps_large_p).item() / np.size(
            np.array(pseudo_label.cpu()))
        pseudo_weight = pseudo_weight * torch.ones(pseudo_label.shape, device=dev)
        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        log_vars['teacher_target_confidence_mean'] = pseudo_prob.mean().item()
        log_vars['teacher_pseudo_label_retain_ratio'] = \
            ps_large_p.float().mean().item()

        if self.enable_strong_aug:
            img, gt_semantic_seg = strong_transform(
                strong_parameters, data=img, target=gt_semantic_seg)
            target_img_src_norm = convert_to_norm(
                target_img, trg_means, trg_stds, src_means, src_stds)
            target_img, _ = strong_transform(
                strong_parameters,
                data=target_img_src_norm,
                target=pseudo_label.unsqueeze(1))
        else:
            target_img = convert_to_norm(
                target_img, trg_means, trg_stds, src_means, src_stds)

        with torch.no_grad():
            ema_src_feat = self.get_ema_model().extract_auxiliary_feat(weak_img)
        mean, covariance, bank = self._update_distribution(
            ema_src_feat, gt_semantic_seg.detach())

        src_mode = 'all' if self.local_iter >= self.start_distribution_iter \
            else 'dec'
        source_losses = self.get_model().forward_train(
            img,
            img_metas,
            gt_semantic_seg,
            mean=mean,
            covariance=covariance,
            bank=bank,
            mode=src_mode)
        source_loss, source_log_vars = self._parse_losses(source_losses)
        log_vars.update(add_prefix(source_log_vars, 'src'))
        source_loss.backward()

        if self.local_iter >= self.start_distribution_iter:
            pseudo_lbl = pseudo_label.clone()
            pseudo_lbl[pseudo_weight == 0.] = self.ignore_index
            target_losses = self.get_model().forward_train(
                target_img,
                target_img_metas,
                pseudo_lbl.unsqueeze(1),
                mean=mean,
                covariance=covariance,
                bank=bank,
                mode='aux')
            target_loss, target_log_vars = self._parse_losses(target_losses)
            log_vars.update(add_prefix(target_log_vars, 'tgt'))
            target_loss.backward()

        local_enable_self_training = self.enable_self_training and (
            not self.push_off_self_training
            or self.local_iter >= self.start_distribution_iter)
        if local_enable_self_training:
            gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mix_masks = get_class_masks(gt_semantic_seg)
            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img[i])),
                    target=torch.stack((gt_semantic_seg[i][0],
                                        pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)
            mix_losses = self.get_model().forward_train(
                mixed_img,
                img_metas,
                mixed_lbl,
                pseudo_weight,
                mode='dec')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(add_prefix(mix_log_vars, 'mix'))
            mix_loss.backward()

        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            grads = [p.grad.detach() for p in params if p.grad is not None]
            log_vars['seg_grad_magnitude'] = calc_grad_magnitude(grads)

        self.local_iter += 1
        return log_vars
