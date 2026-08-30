import torch
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.models import UDA
from mmseg.models.uda.phy_reliability import (PhyReliabilityHead,
                                              masked_bce_loss)
from mmseg.models.uda.proto_gate import ProtoGate
from mmseg.models.uda.struct_prior_gate import StructPriorConsistencyGate
from mmseg.models.uda.uda_decorator import UDADecorator


@UDA.register_module()
class PSSALSupervised(UDADecorator):
    """Supervised PSSAL wrapper without EMA teacher or target self-training."""

    def __init__(self, **cfg):
        super(PSSALSupervised, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.phy_cfg = cfg.get('phy_cfg', dict(enable=False))
        self.enable_phy = self.phy_cfg.get('enable', False)
        self.struct_cfg = cfg.get('struct_cfg', dict(enable=False))
        self.enable_struct = self.struct_cfg.get('enable', False)
        self.proto_cfg = cfg.get('proto_cfg', dict(enable=False))
        self.enable_proto = self.proto_cfg.get('enable', False)
        self.struct_edge_cfg = cfg.get('struct_edge_cfg', dict(enable=False))
        self.enable_struct_edge = self.struct_edge_cfg.get('enable', False)

        if self.enable_phy:
            self.phy_head = PhyReliabilityHead(
                **self.phy_cfg.get('phy_head', {}))
            self.phy_positive_classes = self.phy_cfg.get(
                'phy_positive_classes', [0, 1])
            self.lambda_phy_src = self.phy_cfg.get('lambda_phy_src', 0.1)
        else:
            self.phy_head = None

        if self.enable_struct:
            struct_gate_cfg = dict(self.struct_cfg)
            struct_gate_cfg.pop('enable', None)
            struct_gate_cfg.pop('use_feature_interaction', None)
            struct_gate_cfg.pop('feature_interaction_warmup_iters', None)
            struct_gate_cfg.pop('feature_interaction_rampup_iters', None)
            self.struct_gate = StructPriorConsistencyGate(
                num_classes=cfg['model']['decode_head']['num_classes'],
                **struct_gate_cfg)
        else:
            self.struct_gate = None

        if self.enable_proto:
            proto_gate_cfg = dict(self.proto_cfg)
            proto_gate_cfg.pop('enable', None)
            self.proto_gate = ProtoGate(**proto_gate_cfg)
        else:
            self.proto_gate = None

        if self.enable_struct_edge:
            self.struct_edge_num_classes = self.struct_edge_cfg.get(
                'num_classes', self.num_classes)
            self.struct_edge_ignore_index = self.struct_edge_cfg.get(
                'ignore_index', 255)
            self.struct_edge_loss_weight_src = self.struct_edge_cfg.get(
                'loss_weight_src', 0.02)
            self.struct_edge_warmup_iters = self.struct_edge_cfg.get(
                'warmup_iters', 5000)
            self.struct_edge_prob_weight = self.struct_edge_cfg.get(
                'prob_edge_weight', 0.4)
            self.struct_edge_img_weight = self.struct_edge_cfg.get(
                'img_edge_weight', 0.4)
            self.struct_edge_pair_weight = self.struct_edge_cfg.get(
                'pair_edge_weight', 0.2)
            self.struct_edge_pair_min_weight = self.struct_edge_cfg.get(
                'boundary_pair_min_weight', 0.60)
            self.register_buffer(
                'struct_edge_pair_prior',
                torch.zeros(self.struct_edge_num_classes,
                            self.struct_edge_num_classes))
            self.register_buffer('struct_edge_pair_prior_ready',
                                 torch.zeros(1, dtype=torch.bool))

    def train_step(self, data_batch, optimizer, **kwargs):
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()
        return dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))

    def _class_mask(self, label, classes):
        class_tensor = torch.tensor(classes, device=label.device)
        return torch.any(label[..., None] == class_tensor, dim=-1)

    def _as_label(self, label):
        if label.dim() == 4:
            label = label[:, 0]
        return label.long()

    def _valid_label(self, label):
        return (label != self.struct_edge_ignore_index) & (label >= 0) & (
            label < self.struct_edge_num_classes)

    def _edge_warmup_ratio(self):
        if self.struct_edge_warmup_iters <= 0:
            return 1.0
        return min(1.0, float(self.local_iter) /
                   float(self.struct_edge_warmup_iters))

    def _img_edge(self, img):
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
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + eps).mean(
            dim=1, keepdim=True)
        return edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps)

    def _prob_edge(self, probs):
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
        return edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps)

    def _boundary_pair_matrix(self, label):
        label = self._as_label(label)
        matrix = label.new_zeros(
            (self.struct_edge_num_classes, self.struct_edge_num_classes),
            dtype=torch.float)
        valid = self._valid_label(label)
        for dy, dx in ((0, 1), (1, 0)):
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
    def _update_pair_prior(self, label):
        pair_matrix = self._boundary_pair_matrix(label).to(
            self.struct_edge_pair_prior)
        if pair_matrix.max() <= 0:
            return
        self.struct_edge_pair_prior.copy_(pair_matrix)
        self.struct_edge_pair_prior_ready.fill_(True)

    def _pair_response(self, label):
        label = self._as_label(label)
        response = label.new_zeros(label.shape, dtype=torch.float)
        if not self.struct_edge_pair_prior_ready.item():
            return response.unsqueeze(1)
        pair_prior = self.struct_edge_pair_prior.to(device=label.device)
        valid = self._valid_label(label)
        has_pair = torch.zeros_like(valid)
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
            if not boundary.any():
                continue
            score = pair_prior[label.clamp(0, self.struct_edge_num_classes - 1),
                               shifted.clamp(0, self.struct_edge_num_classes - 1)]
            response = torch.maximum(
                response, torch.where(boundary, score,
                                      response.new_zeros(())))
            has_pair |= boundary
        response = torch.where(has_pair, response, response.new_zeros(()))
        return response.unsqueeze(1).clamp(0.0, 1.0)

    def _struct_edge_weight(self, img, label, probs, log_vars):
        if not self.enable_struct_edge:
            return None
        self._update_pair_prior(label)
        img_edge = self._img_edge(img)
        prob_edge = self._prob_edge(probs)
        pair_response = self._pair_response(label)
        gate = (
            self.struct_edge_prob_weight * prob_edge +
            self.struct_edge_img_weight * img_edge +
            self.struct_edge_pair_weight * pair_response).clamp(0.0, 1.0)
        weight = gate * (self.struct_edge_loss_weight_src *
                         self._edge_warmup_ratio())
        log_vars['prob_edge_src_mean'] = prob_edge.detach().mean().item()
        log_vars['img_edge_src_mean'] = img_edge.detach().mean().item()
        log_vars['boundary_pair_response_src_mean'] = \
            pair_response.detach().mean().item()
        log_vars['struct_edge_w_src_mean'] = weight.detach().mean().item()
        return weight.detach()

    def _compute_phy_loss(self, vi_img, gt, log_vars):
        r_phy = self.phy_head(vi_img.float())
        label = self._as_label(gt)
        target = self._class_mask(
            label, self.phy_positive_classes).unsqueeze(1)
        valid = label.ne(255).unsqueeze(1)
        loss_phy = masked_bce_loss(r_phy, target, valid)
        log_vars['loss_phy_src'] = loss_phy.detach().item()
        log_vars['R_phy_mean'] = r_phy.detach().mean().item()
        return self.lambda_phy_src * loss_phy

    def _student_probs(self, img, img_metas):
        dropout_states = []
        for m in self.get_model().modules():
            if isinstance(m, _DropoutNd):
                dropout_states.append((m, m.training))
                m.training = False
            if isinstance(m, DropPath):
                dropout_states.append((m, m.training))
                m.training = False
        try:
            with torch.no_grad():
                logits = self.get_model().generate_pseudo_label(
                    img, img_metas)
        finally:
            for module, was_training in dropout_states:
                module.training = was_training
        return torch.softmax(logits.detach(), dim=1)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      vi_img=None,
                      **kwargs):
        del kwargs
        log_vars = {}
        label = self._as_label(gt_semantic_seg)
        probs_detached = self._student_probs(img, img_metas)
        struct_prior_response = None
        if self.enable_struct:
            pseudo_weight = torch.ones_like(label, dtype=torch.float)
            _, struct_debug = self.struct_gate.apply(
                target_probs=probs_detached,
                pseudo_label=label,
                pseudo_weight=pseudo_weight,
                source_gt=gt_semantic_seg,
                source_probs=probs_detached,
                iter=self.local_iter,
                ignore_index=255)
            log_vars['R_struct_mean'] = \
                struct_debug['R_struct'].detach().mean().item()
            if self.struct_cfg.get('use_feature_interaction', False):
                feat_warmup = self.struct_cfg.get(
                    'feature_interaction_warmup_iters',
                    self.struct_cfg.get('struct_warmup_iters', 0))
                feat_rampup = self.struct_cfg.get(
                    'feature_interaction_rampup_iters',
                    self.struct_cfg.get('struct_rampup_iters', 1))
                if self.local_iter >= feat_warmup:
                    feat_ratio = min(
                        1.0,
                        max(0.0, (self.local_iter - feat_warmup) /
                            max(float(feat_rampup), 1.0)))
                    struct_prior_response = \
                        struct_debug['R_struct'].unsqueeze(1) * feat_ratio
                    log_vars['struct_feature_ratio'] = feat_ratio
        struct_edge_weight = self._struct_edge_weight(
            img, gt_semantic_seg, probs_detached, log_vars)
        losses = self.get_model().forward_train(
            img,
            img_metas,
            gt_semantic_seg,
            struct_edge_weight=struct_edge_weight,
            struct_prior_response=struct_prior_response,
            return_feat=True)
        feat = losses.pop('features')
        loss, parsed = self._parse_losses(losses)
        log_vars.update(parsed)

        if self.enable_phy:
            if vi_img is None:
                raise ValueError('PSSALSupervised phy_cfg requires vi_img.')
            loss_phy = self._compute_phy_loss(vi_img, gt_semantic_seg,
                                              log_vars)
            loss = loss + loss_phy

        if self.enable_proto:
            proto_src_log = self.proto_gate.update_source(
                feat, gt_semantic_seg, iter=self.local_iter)
            if 'update_ratio' in proto_src_log:
                log_vars['proto_src_update_ratio'] = \
                    proto_src_log['update_ratio'].detach().item()
            final_weight = torch.ones_like(label, dtype=torch.float)
            loss_proto, _ = self.proto_gate.loss_target(
                feat, label, final_weight, iter=self.local_iter)
            loss_proto = self.proto_cfg.get('lambda_proto_src', 0.001) * \
                loss_proto
            log_vars['loss_proto_src'] = loss_proto.detach().item()
            loss = loss + loss_proto

        log_vars['loss'] = loss.detach().item()
        loss.backward()

        self.local_iter += 1
        return log_vars
