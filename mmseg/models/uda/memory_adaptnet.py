# Adapted from MemoryAdaptNet:
# https://github.com/ChenjieCSU/MemoryAdaptNet-master
# Code is provided by the original authors for non-commercial research use.

import math

import torch
import torch.nn.functional as F
from torch import nn, optim

from mmseg.models import UDA
from mmseg.models.discriminators.fc_discriminator import FCDiscriminator
from mmseg.models.uda.uda_decorator import UDADecorator
from mmseg.ops import resize


@UDA.register_module()
class MemoryAdaptNet(UDADecorator):
    """UDA wrapper for MemoryAdaptNet-R101 + 3VI."""

    def __init__(self, **cfg):
        super(MemoryAdaptNet, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.num_classes = cfg.get('num_classes', self.num_classes)
        self.ignore_index = cfg.get('ignore_index', 255)
        self.memory_start_iter = cfg.get('memory_start_iter', 3000)
        self.target_entropy_threshold = cfg.get('target_entropy_threshold',
                                                0.3)
        self.entropy_eps = cfg.get('entropy_eps', 1e-30)
        self.lambda_stage1 = cfg.get('lambda_stage1', 1.0)
        self.lambda_stage2 = cfg.get('lambda_stage2', 1.0)
        self.lambda_adv = cfg.get('lambda_adv', 0.001)
        self.source_domain_label = cfg.get('source_domain_label', 0.0)
        self.target_domain_label = cfg.get('target_domain_label', 1.0)
        self.lr_d_power = cfg.get('lr_d_power', 1.0)
        self.lr_d_min = cfg.get('lr_d_min', 0.0)

        discriminator_cfg = dict(cfg['discriminator'])
        discriminator_cfg.pop('type', None)
        self.discriminator = FCDiscriminator(**discriminator_cfg)
        self.loss_fn_d = nn.BCEWithLogitsLoss()
        opt_cfg = dict(cfg['discriminator_optimizer'])
        opt_type = opt_cfg.pop('type')
        assert opt_type == 'Adam'
        self.optimizer_d = optim.Adam(self.discriminator.parameters(),
                                      **opt_cfg)
        self.base_lr_d = opt_cfg['lr']

    def _memory_enabled(self):
        return self.local_iter >= self.memory_start_iter

    def _set_discriminator_grad(self, requires_grad):
        for param in self.discriminator.parameters():
            param.requires_grad = requires_grad

    def _adjust_discriminator_lr(self):
        coeff = (1 - self.local_iter / self.max_iters)**self.lr_d_power
        lr = (self.base_lr_d - self.lr_d_min) * coeff + self.lr_d_min
        for group in self.optimizer_d.param_groups:
            group['lr'] = lr

    def _domain_label_like(self, pred, label):
        return pred.new_full(pred.size(), label)

    def _stage_loss(self, logits, labels, image_shape, weight, name):
        logits = resize(
            logits,
            size=image_shape,
            mode='bilinear',
            align_corners=False)
        return {
            name:
            weight * F.cross_entropy(
                logits, labels.squeeze(1).long(), ignore_index=self.ignore_index)
        }

    def _target_filtered_pseudo(self, logits):
        prob = F.softmax(logits.detach(), dim=1)
        conf, pseudo = prob.max(dim=1)
        entropy = -(prob * torch.log(prob + self.entropy_eps)).sum(dim=1)
        entropy = entropy / math.log(self.num_classes)
        pseudo = pseudo.clone()
        pseudo[entropy > self.target_entropy_threshold] = self.ignore_index
        return pseudo, conf, entropy

    def train_step(self, data_batch, optimizer, **kwargs):
        optimizer.zero_grad()
        self.optimizer_d.zero_grad()
        self._adjust_discriminator_lr()

        log_vars = self(**data_batch)

        optimizer.step()
        self.optimizer_d.step()

        log_vars.pop('loss', None)
        self.local_iter += 1
        return dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        use_memory = self._memory_enabled()
        self._set_discriminator_grad(False)

        src = self.model.forward_memory(img, use_memory=use_memory)
        tgt = self.model.forward_memory(target_img, use_memory=use_memory)

        student_losses = {}
        student_losses.update(
            self._stage_loss(src['stage1_logits'], gt_semantic_seg,
                             img.shape[2:], self.lambda_stage1,
                             'loss_seg_stage1'))
        if use_memory:
            student_losses.update(
                self._stage_loss(src['stage2_logits'], gt_semantic_seg,
                                 img.shape[2:], self.lambda_stage2,
                                 'loss_seg_stage2'))
            self.model.memory_module.update(
                src['memory_features'],
                gt_semantic_seg,
                ignore_index=self.ignore_index,
                learning_rate=self._current_seg_lr(),
                allow_uninitialized=True)
            target_pseudo, _, _ = self._target_filtered_pseudo(
                resize(
                    tgt['stage2_logits'],
                    size=target_img.shape[2:],
                    mode='bilinear',
                    align_corners=False))
            self.model.memory_module.update(
                tgt['memory_features'],
                target_pseudo,
                ignore_index=self.ignore_index,
                learning_rate=self._current_seg_lr(),
                allow_uninitialized=False)

        target_prob = F.softmax(tgt['stage1_logits'], dim=1)
        out_tgt = self.discriminator(target_prob)
        student_losses['loss_adv_target'] = self.lambda_adv * self.loss_fn_d(
            out_tgt, self._domain_label_like(out_tgt,
                                             self.source_domain_label))
        student_loss, student_log_vars = self._parse_losses(student_losses)
        student_loss.backward()

        self._set_discriminator_grad(True)
        source_prob = F.softmax(src['stage1_logits'].detach(), dim=1)
        target_prob = F.softmax(tgt['stage1_logits'].detach(), dim=1)
        out_src = self.discriminator(source_prob)
        out_tgt = self.discriminator(target_prob)
        d_losses = dict(
            loss_d_source=0.5 * self.loss_fn_d(
                out_src,
                self._domain_label_like(out_src, self.source_domain_label)),
            loss_d_target=0.5 * self.loss_fn_d(
                out_tgt,
                self._domain_label_like(out_tgt, self.target_domain_label)))
        d_loss, d_log_vars = self._parse_losses(d_losses)
        d_loss.backward()

        return {
            **student_log_vars,
            **d_log_vars,
            'memory_enabled': float(use_memory)
        }

    def _current_seg_lr(self):
        return None

    def memoryadaptnet_optimizer_state_dict(self):
        return dict(
            local_iter=self.local_iter,
            discriminator=self.optimizer_d.state_dict())

    def load_memoryadaptnet_optimizer_state_dict(self, state_dict):
        self.local_iter = state_dict.get('local_iter', self.local_iter)
        if 'discriminator' in state_dict:
            self.optimizer_d.load_state_dict(state_dict['discriminator'])
