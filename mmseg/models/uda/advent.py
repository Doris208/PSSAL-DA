# Adapted from ADVENT:
# https://github.com/valeoai/ADVENT
# Licensed under the Apache License, Version 2.0.

import torch
from torch import nn, optim

from mmseg.models import UDA
from mmseg.models.discriminators.fc_discriminator import FCDiscriminator
from mmseg.models.uda.uda_decorator import UDADecorator
from mmseg.models.utils.advent_entropy import prob_2_entropy


@UDA.register_module()
class AdvEnt(UDADecorator):
    """Output-space adversarial entropy minimization for UDA.

    The student segmentation model is optimized with source supervision and a
    target adversarial objective. The discriminators are optimized separately
    on detached source/target weighted self-information maps.
    """

    def __init__(self, **cfg):
        super(AdvEnt, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.num_classes = cfg.get('num_classes', self.num_classes)
        self.ignore_index = cfg.get('ignore_index', 255)
        self.multi_level = cfg.get('multi_level', True)
        self.lambda_seg_main = cfg.get('lambda_seg_main', 1.0)
        self.lambda_seg_aux = cfg.get('lambda_seg_aux', 0.1)
        self.lambda_adv_main = cfg.get('lambda_adv_main', 0.001)
        self.lambda_adv_aux = cfg.get('lambda_adv_aux', 0.0002)
        self.source_domain_label = cfg.get('source_domain_label', 0.0)
        self.target_domain_label = cfg.get('target_domain_label', 1.0)
        self.entropy_eps = cfg.get('entropy_eps', 1e-30)
        self.normalize_entropy = cfg.get('normalize_entropy', True)
        self.lr_d_power = cfg.get('lr_d_power', 1.0)
        self.lr_d_min = cfg.get('lr_d_min', 0.0)

        assert self.multi_level
        assert self.model.with_auxiliary_head

        discriminator_cfg = dict(cfg['discriminator'])
        discriminator_cfg.pop('type', None)
        self.d_main = FCDiscriminator(**discriminator_cfg)
        self.d_aux = FCDiscriminator(**discriminator_cfg)
        self.loss_fn_d = nn.BCEWithLogitsLoss()

        opt_cfg = dict(cfg['discriminator_optimizer'])
        opt_type = opt_cfg.pop('type')
        assert opt_type == 'Adam'
        self.optimizer_d_main = optim.Adam(self.d_main.parameters(), **opt_cfg)
        self.optimizer_d_aux = optim.Adam(self.d_aux.parameters(), **opt_cfg)
        self.base_lr_d_main = opt_cfg['lr']
        self.base_lr_d_aux = opt_cfg['lr']

    def _set_discriminator_grad(self, requires_grad):
        for discriminator in (self.d_main, self.d_aux):
            for param in discriminator.parameters():
                param.requires_grad = requires_grad

    def _adjust_discriminator_lr(self, optimizer, base_lr):
        coeff = (1 - self.local_iter / self.max_iters)**self.lr_d_power
        lr = (base_lr - self.lr_d_min) * coeff + self.lr_d_min
        for group in optimizer.param_groups:
            group['lr'] = lr

    def _domain_label_like(self, pred, label):
        return pred.new_full(pred.size(), label)

    def _entropy(self, logits):
        return prob_2_entropy(
            logits,
            num_classes=self.num_classes,
            eps=self.entropy_eps,
            normalize=self.normalize_entropy)

    def _seg_loss(self, logits, gt_semantic_seg, head, prefix, weight):
        raw_losses = head.losses(logits, gt_semantic_seg)
        losses = {}
        for name, value in raw_losses.items():
            if name == 'loss_seg':
                losses[f'loss_seg_{prefix}'] = value * weight
            elif name.startswith('acc_'):
                losses[f'{prefix}_{name}'] = value
        return losses

    def train_step(self, data_batch, optimizer, **kwargs):
        optimizer.zero_grad()
        self.optimizer_d_main.zero_grad()
        self.optimizer_d_aux.zero_grad()
        self._adjust_discriminator_lr(self.optimizer_d_main,
                                      self.base_lr_d_main)
        self._adjust_discriminator_lr(self.optimizer_d_aux,
                                      self.base_lr_d_aux)

        log_vars = self(**data_batch)

        optimizer.step()
        self.optimizer_d_main.step()
        self.optimizer_d_aux.step()

        log_vars.pop('loss', None)
        self.local_iter += 1
        return dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        self._set_discriminator_grad(False)

        pred_src = self.model.forward_with_aux(img, img_metas)
        pred_tgt = self.model.forward_with_aux(target_img, target_img_metas)
        assert 'main' in pred_src and 'aux' in pred_src
        assert 'main' in pred_tgt and 'aux' in pred_tgt

        student_losses = {}
        student_losses.update(
            self._seg_loss(pred_src['main'], gt_semantic_seg,
                           self.model.decode_head, 'main',
                           self.lambda_seg_main))
        student_losses.update(
            self._seg_loss(pred_src['aux'], gt_semantic_seg,
                           self.model.auxiliary_head, 'aux',
                           self.lambda_seg_aux))

        entropy_tgt_main = self._entropy(pred_tgt['main'])
        entropy_tgt_aux = self._entropy(pred_tgt['aux'])
        out_tgt_main = self.d_main(entropy_tgt_main)
        out_tgt_aux = self.d_aux(entropy_tgt_aux)
        student_losses['loss_adv_main'] = self.lambda_adv_main * self.loss_fn_d(
            out_tgt_main,
            self._domain_label_like(out_tgt_main, self.source_domain_label))
        student_losses['loss_adv_aux'] = self.lambda_adv_aux * self.loss_fn_d(
            out_tgt_aux,
            self._domain_label_like(out_tgt_aux, self.source_domain_label))

        student_loss, student_log_vars = self._parse_losses(student_losses)
        student_loss.backward()

        self._set_discriminator_grad(True)

        entropy_src_main = self._entropy(pred_src['main']).detach()
        entropy_src_aux = self._entropy(pred_src['aux']).detach()
        entropy_tgt_main = entropy_tgt_main.detach()
        entropy_tgt_aux = entropy_tgt_aux.detach()

        d_losses = {}
        out_src_main = self.d_main(entropy_src_main)
        out_tgt_main = self.d_main(entropy_tgt_main)
        out_src_aux = self.d_aux(entropy_src_aux)
        out_tgt_aux = self.d_aux(entropy_tgt_aux)

        d_losses['loss_d_main_src'] = 0.5 * self.loss_fn_d(
            out_src_main,
            self._domain_label_like(out_src_main, self.source_domain_label))
        d_losses['loss_d_main_tgt'] = 0.5 * self.loss_fn_d(
            out_tgt_main,
            self._domain_label_like(out_tgt_main, self.target_domain_label))
        d_losses['loss_d_aux_src'] = 0.5 * self.loss_fn_d(
            out_src_aux,
            self._domain_label_like(out_src_aux, self.source_domain_label))
        d_losses['loss_d_aux_tgt'] = 0.5 * self.loss_fn_d(
            out_tgt_aux,
            self._domain_label_like(out_tgt_aux, self.target_domain_label))

        d_loss, d_log_vars = self._parse_losses(d_losses)
        d_loss.backward()

        return {**student_log_vars, **d_log_vars}

    def advent_optimizer_state_dict(self):
        return dict(
            local_iter=self.local_iter,
            d_main=self.optimizer_d_main.state_dict(),
            d_aux=self.optimizer_d_aux.state_dict())

    def load_advent_optimizer_state_dict(self, state_dict):
        self.local_iter = state_dict.get('local_iter', self.local_iter)
        if 'd_main' in state_dict:
            self.optimizer_d_main.load_state_dict(state_dict['d_main'])
        if 'd_aux' in state_dict:
            self.optimizer_d_aux.load_state_dict(state_dict['d_aux'])
