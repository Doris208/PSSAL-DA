# Adapted from ProCA for the local HRDA/MMSegmentation training stack.

import os
from copy import deepcopy

import mmcv
import torch
import torch.nn.functional as F
from PIL import Image

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.losses.proca_prototype_contrastive_loss import \
    proca_prototype_contrastive_loss
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.proca_prototype_estimator import \
    ProCAPrototypeEstimator


@UDA.register_module()
class ProCA(UDADecorator):

    def __init__(self, **cfg):
        super(ProCA, self).__init__(**cfg)
        self.local_iter = 0
        self.num_classes = cfg.get('num_classes',
                                   cfg['model']['decode_head']['num_classes'])
        self.ignore_index = cfg.get('ignore_index', 255)
        self.feature_index = cfg.get('feature_index', 3)
        self.feature_dim = cfg.get('feature_dim', 2048)
        self.multi_level = cfg.get('multi_level', True)
        self.target_pseudo_threshold = cfg.get('target_pseudo_threshold', 0.9)
        self.temperature = cfg.get('temperature', 50.0)
        self.lambda_feat = cfg.get('lambda_feat', 1.0)
        self.lambda_out = cfg.get('lambda_out', 1.0)
        self.lambda_ssl = cfg.get('lambda_ssl', 1.0)
        self.use_momentum = cfg.get('use_momentum', False)
        self.prototype_momentum = cfg.get('prototype_momentum', 0.9)
        self.proca_stage = cfg.get('proca_stage', 'adapt')
        self.ssl_pseudo_label_root = cfg.get('ssl_pseudo_label_root', None)

        self.feature_estimator = ProCAPrototypeEstimator(
            self.num_classes,
            self.feature_dim,
            self.ignore_index,
            self.use_momentum,
            self.prototype_momentum)
        self.output_estimator = ProCAPrototypeEstimator(
            self.num_classes,
            self.num_classes,
            self.ignore_index,
            self.use_momentum,
            self.prototype_momentum)

        init_feat = cfg.get('init_feature_prototype', None)
        init_out = cfg.get('init_output_prototype', None)
        if init_feat is not None:
            self.feature_estimator.load_distribution(init_feat)
        if init_out is not None:
            self.output_estimator.load_distribution(init_out)

    def get_model(self):
        return get_module(self.model)

    def load_state_dict(self, state_dict, strict=False):
        mapped = {}
        for key, value in state_dict.items():
            clean_key = key[7:] if key.startswith('module.') else key
            if clean_key.startswith(('backbone.', 'decode_head.', 'neck.')):
                mapped['model.' + clean_key] = value
            else:
                mapped[clean_key] = value
        return super(ProCA, self).load_state_dict(mapped, strict=False)

    def train_step(self, data_batch, optimizer, **kwargs):
        optimizer.zero_grad()
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)
        loss.backward()
        optimizer.step()
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def _extract_feature_and_logits(self, img, img_metas):
        feats = self.get_model().extract_feat(img)
        feat = feats[self.feature_index] if isinstance(feats, (list, tuple)) \
            else feats
        logits = self.get_model().decode_head.forward_test(
            feats, img_metas, self.get_model().test_cfg)
        return feats, feat, logits

    def _pseudo_from_logits(self, logits):
        prob = torch.softmax(logits.detach(), dim=1)
        conf, label = prob.max(dim=1)
        pseudo = label.clone()
        pseudo[conf < self.target_pseudo_threshold] = self.ignore_index
        return pseudo.unsqueeze(1), conf

    def _prototype_losses(self, feat, logits, labels, prefix):
        losses = {}
        losses[f'{prefix}.loss_feat_pcl'] = \
            self.lambda_feat * proca_prototype_contrastive_loss(
                feat, labels, self.feature_estimator.prototype,
                self.temperature, self.ignore_index)
        if self.multi_level:
            losses[f'{prefix}.loss_out_pcl'] = \
                self.lambda_out * proca_prototype_contrastive_loss(
                    logits, labels, self.output_estimator.prototype,
                    self.temperature, self.ignore_index)
        return losses

    def _update_prototypes(self, feat, logits, labels, prefix):
        feat_counts = self.feature_estimator.update(feat, labels)
        out_counts = self.output_estimator.update(logits, labels)
        return {
            f'{prefix}.feat_proto_updated_classes':
            (feat_counts > 0).float().sum(),
            f'{prefix}.out_proto_updated_classes':
            (out_counts > 0).float().sum(),
        }

    def _pseudo_path(self, img_meta):
        rel = img_meta.get('ori_filename', img_meta.get('filename', None))
        if rel is None:
            raise KeyError('img_meta must contain ori_filename or filename.')
        rel = rel.replace('\\', '/')
        if '/img_dir/' in rel:
            rel = rel.split('/img_dir/', 1)[1]
            rel = rel.split('/', 1)[1] if '/' in rel else rel
        rel = rel.lstrip('/')
        return os.path.join(self.ssl_pseudo_label_root, rel)

    def _load_ssl_labels(self, target_img, target_img_metas):
        if self.ssl_pseudo_label_root is None:
            raise ValueError('ssl_pseudo_label_root is required for SSL stage.')
        labels = []
        for meta in target_img_metas:
            path = self._pseudo_path(meta)
            if not os.path.exists(path):
                raise FileNotFoundError(f'Missing ProCA pseudo label: {path}')
            label = mmcv.imread(path, flag='unchanged', backend='pillow')
            label = torch.from_numpy(label).to(
                target_img.device, dtype=torch.long)
            labels.append(label.unsqueeze(0).unsqueeze(0))
        labels = torch.cat(labels, dim=0)
        if labels.shape[-2:] != target_img.shape[-2:]:
            labels = F.interpolate(
                labels.float(), size=target_img.shape[-2:],
                mode='nearest').long()
        return labels

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      target_gt_semantic_seg=None,
                      **kwargs):
        if self.proca_stage == 'ssl':
            pseudo = self._load_ssl_labels(target_img, target_img_metas)
            losses = self.get_model().forward_train(
                target_img, target_img_metas, pseudo)
            return add_prefix(losses, 'ssl')

        losses = {}
        source_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg)
        losses.update(add_prefix(source_losses, 'src'))

        _, src_feat, src_logits = self._extract_feature_and_logits(
            img, img_metas)
        with torch.no_grad():
            self._update_prototypes(src_feat, src_logits.detach(),
                                    gt_semantic_seg, 'src')
        losses.update(
            self._prototype_losses(src_feat, src_logits, gt_semantic_seg,
                                   'src'))

        _, tgt_feat, tgt_logits = self._extract_feature_and_logits(
            target_img, target_img_metas)
        target_pseudo, target_conf = self._pseudo_from_logits(
            self.get_model().encode_decode(target_img, target_img_metas))
        valid = target_pseudo != self.ignore_index
        losses['target_pseudo_retain_ratio'] = valid.float().mean()
        losses['target_pseudo_confidence_mean'] = target_conf.mean()
        with torch.no_grad():
            self._update_prototypes(tgt_feat, tgt_logits.detach(),
                                    target_pseudo, 'tgt')
        losses.update(
            self._prototype_losses(tgt_feat, tgt_logits, target_pseudo, 'tgt'))

        target_ssl_losses = self.get_model().forward_train(
            target_img, target_img_metas, target_pseudo)
        for key, value in add_prefix(target_ssl_losses, 'tgt_ssl').items():
            losses[key] = value * self.lambda_ssl

        self.local_iter += 1
        return losses
