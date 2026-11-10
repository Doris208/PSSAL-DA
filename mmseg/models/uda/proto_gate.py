import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoGate(nn.Module):
    """Prototype-guided semantic consistency refinement for pseudo labels."""

    def __init__(self,
                 num_classes=5,
                 ignore_index=255,
                 momentum=0.999,
                 min_pixels=16,
                 source_update=True,
                 target_update=True,
                 target_update_start=3000,
                 target_conf_thr=None,
                 target_weight_thr=None,
                 proto_min_weight=0.80,
                 proto_temp=0.10,
                 proto_margin_thr=0.02,
                 proto_power=0.50,
                 apply_to_classes=(0, 1, 2, 3, 4),
                 weight_start=0,
                 min_valid_classes_for_weight=0,
                 lambda_proto_tgt=0.005,
                 lambda_proto_src=0.0,
                 loss_start=3000,
                 min_valid_classes_for_loss=0,
                 save_proto_debug=True,
                 proto_debug_interval=1500,
                 log_class_stats=True,
                 detach=True,
                 separate_domain_prototypes=False,
                 prototype_fusion_weight=0.5,
                 log_stats=True,
                 collect_debug=True,
                 tau_proto=None,
                 tau_conf_proto=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.momentum = momentum
        self.min_pixels = min_pixels
        self.source_update = source_update
        self.target_update = target_update
        self.target_update_start = target_update_start
        for name, value, alias in (('tau_proto', tau_proto, target_weight_thr),
                                   ('tau_conf_proto', tau_conf_proto, target_conf_thr)):
            if value is not None and alias is not None and value != alias:
                raise ValueError(name + ' conflicts with its legacy threshold alias')
        self.tau_proto = (tau_proto if tau_proto is not None else
                          target_weight_thr if target_weight_thr is not None else 0.70)
        self.tau_conf_proto = (tau_conf_proto if tau_conf_proto is not None else
                               target_conf_thr if target_conf_thr is not None else 0.90)
        self.target_conf_thr = self.tau_conf_proto
        self.target_weight_thr = self.tau_proto
        self.collect_diagnostics = False
        self.diagnostic_spatial_mask = None
        self.last_target_diagnostics = {}
        self.proto_min_weight = proto_min_weight
        self.proto_temp = proto_temp
        self.proto_margin_thr = proto_margin_thr
        self.proto_power = proto_power
        self.apply_to_classes = list(apply_to_classes)
        self.weight_start = weight_start
        self.min_valid_classes_for_weight = min_valid_classes_for_weight
        self.lambda_proto_tgt = lambda_proto_tgt
        self.lambda_proto_src = lambda_proto_src
        self.loss_start = loss_start
        self.min_valid_classes_for_loss = min_valid_classes_for_loss
        self.save_proto_debug = save_proto_debug
        self.proto_debug_interval = proto_debug_interval
        self.log_class_stats = log_class_stats
        self.detach = detach
        self.separate_domain_prototypes = separate_domain_prototypes
        self.prototype_fusion_weight = prototype_fusion_weight
        self.log_stats = log_stats
        self.collect_debug = collect_debug

        self.register_buffer('prototypes', torch.empty(0))
        self.register_buffer('proto_counts', torch.zeros(num_classes))
        self.register_buffer('valid_proto', torch.zeros(num_classes,
                                                        dtype=torch.bool))
        self.register_buffer('source_prototypes', torch.empty(0))
        self.register_buffer('target_prototypes', torch.empty(0))
        self.register_buffer('source_valid_proto',
                             torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('target_valid_proto',
                             torch.zeros(num_classes, dtype=torch.bool))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Checkpoints contain populated banks; a fresh module starts with empty
        # buffers. Allocate the saved shape before the normal loader copies it.
        for name in ('prototypes', 'source_prototypes', 'target_prototypes'):
            saved = state_dict.get(prefix + name)
            current = getattr(self, name)
            if saved is not None and current.numel() == 0 and saved.ndim == 2 \
                    and saved.shape[0] == self.num_classes:
                setattr(self, name, torch.empty(saved.shape, device=current.device,
                                               dtype=current.dtype))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _select_feat(self, feat):
        if isinstance(feat, dict):
            for key in ('features', 'feat', 'x'):
                if key in feat:
                    feat = feat[key]
                    break
            else:
                feat = next(iter(feat.values()))
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        return feat

    def _init_proto(self, feat):
        feat = self._select_feat(feat)
        if self.prototypes.numel() == 0:
            self.prototypes = feat.new_zeros(self.num_classes, feat.shape[1])
            self.proto_counts = feat.new_zeros(self.num_classes)
            self.valid_proto = torch.zeros(
                self.num_classes, dtype=torch.bool, device=feat.device)
        if self.separate_domain_prototypes and \
                self.source_prototypes.numel() == 0:
            self.source_prototypes = feat.new_zeros(
                self.num_classes, feat.shape[1])
            self.target_prototypes = feat.new_zeros(
                self.num_classes, feat.shape[1])
            self.source_valid_proto = torch.zeros(
                self.num_classes, dtype=torch.bool, device=feat.device)
            self.target_valid_proto = torch.zeros(
                self.num_classes, dtype=torch.bool, device=feat.device)

    def _resize_label(self, label, size):
        if label.dim() == 4:
            label = label[:, 0]
        if label.shape[-2:] == size:
            return label.long()
        return F.interpolate(
            label.unsqueeze(1).float(), size=size,
            mode='nearest').squeeze(1).long()

    def _resize_map(self, value, size):
        if value.dim() == 4:
            value = value[:, 0]
        if value.shape[-2:] == size:
            return value.float()
        return F.interpolate(
            value.unsqueeze(1).float(), size=size, mode='bilinear',
            align_corners=False).squeeze(1)

    def _empty_update_log(self, feat):
        if not self.log_stats:
            return {}
        dev = self._select_feat(feat).device
        log = dict(update_ratio=torch.tensor(0.0, device=dev))
        if self.log_class_stats:
            for cls in range(self.num_classes):
                log[f'update_class_{cls}_ratio'] = torch.tensor(0.0,
                                                                device=dev)
        return log

    @torch.no_grad()
    def _update_from_mask(self, feat, label, valid, domain=None):
        self._init_proto(feat)
        feat = self._select_feat(feat).detach()
        label = self._resize_label(label, feat.shape[-2:])
        valid = valid & label.ne(self.ignore_index)
        total = valid.numel()
        updated = feat.new_zeros(()) if self.log_stats else None
        class_ratios = []
        flat_feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        flat_label = label.reshape(-1)
        flat_valid = valid.reshape(-1)
        for cls in range(self.num_classes):
            cls_mask = flat_valid & flat_label.eq(cls)
            cls_pixels = cls_mask.sum()
            if self.log_stats and self.log_class_stats:
                class_ratios.append(cls_pixels.float() / max(float(total), 1.0))
            if cls_pixels.item() < self.min_pixels:
                continue
            new_proto = flat_feat[cls_mask].mean(dim=0)
            new_proto = F.normalize(new_proto, dim=0)
            if self.separate_domain_prototypes and domain in ('source',
                                                              'target'):
                if domain == 'source':
                    bank = self.source_prototypes
                    valid_bank = self.source_valid_proto
                else:
                    bank = self.target_prototypes
                    valid_bank = self.target_valid_proto
                if not valid_bank[cls]:
                    bank[cls] = new_proto
                    valid_bank[cls] = True
                else:
                    bank[cls] = self.momentum * bank[cls] + (
                        1.0 - self.momentum) * new_proto
                    bank[cls] = F.normalize(bank[cls], dim=0)
            elif not self.valid_proto[cls]:
                self.prototypes[cls] = new_proto
                self.valid_proto[cls] = True
            else:
                self.prototypes[cls] = (
                    self.momentum * self.prototypes[cls] +
                    (1.0 - self.momentum) * new_proto)
                self.prototypes[cls] = F.normalize(
                    self.prototypes[cls], dim=0)
            self.proto_counts[cls] += cls_pixels.float()
            if self.log_stats:
                updated = updated + cls_pixels.float()
        if self.separate_domain_prototypes:
            self._refresh_fused_prototypes()
        if not self.log_stats:
            return {}
        log = dict(update_ratio=updated / max(float(total), 1.0))
        if self.log_class_stats:
            for cls, ratio in enumerate(class_ratios):
                log[f'update_class_{cls}_ratio'] = ratio
        return log

    def update_source(self, feat, gt, iter=None):
        if not self.source_update:
            return self._empty_update_log(feat)
        feat_sel = self._select_feat(feat)
        label = self._resize_label(gt, feat_sel.shape[-2:])
        valid = label.ne(self.ignore_index)
        return self._update_from_mask(feat_sel, label, valid, domain='source')

    def update_target(self, feat, pseudo_label, pseudo_conf, pseudo_weight,
                      iter=None):
        update_active = (self.target_update and iter is not None and
                         iter >= self.target_update_start)
        self.last_target_diagnostics = {}
        if not update_active and not self.collect_diagnostics:
            return self._empty_update_log(feat)
        feat_sel = self._select_feat(feat)
        size = feat_sel.shape[-2:]
        label = self._resize_label(pseudo_label, size)
        conf = self._resize_map(pseudo_conf, size)
        weight = self._resize_map(pseudo_weight, size)
        valid = conf.ge(self.tau_conf_proto) & weight.ge(
            self.tau_proto) & label.ne(self.ignore_index)
        if self.collect_diagnostics:
            diagnostics = self.last_target_diagnostics
            diagnostics['target_update_active'] = float(update_active)
            support = label.ne(self.ignore_index)
            if self.diagnostic_spatial_mask is not None:
                # Same bilinear support as w_t_ps: exclude wholly invalid cells,
                # retain partial support that can genuinely pass the threshold.
                support = support & self._resize_map(
                    self.diagnostic_spatial_mask.float(), size).gt(0)
            for cls in range(self.num_classes):
                candidate = label.eq(cls) & support
                selected = candidate & valid
                count = selected.sum()
                prefix = 'class_{}.'.format(cls)
                diagnostics[prefix + 'candidate_ratio'] = candidate.float().mean()
                denom = candidate.sum().clamp_min(1)
                diagnostics[prefix + 'conf_pass_ratio'] = (
                    candidate & conf.ge(self.tau_conf_proto)).sum().float() / denom
                diagnostics[prefix + 'confidence_pass_count'] = (
                    candidate & conf.ge(self.tau_conf_proto)).sum()
                diagnostics[prefix + 'w_t_ps_pass_ratio'] = (
                    candidate & weight.ge(self.tau_proto)).sum().float() / denom
                diagnostics[prefix + 'weight_pass_count'] = (
                    candidate & weight.ge(self.tau_proto)).sum()
                diagnostics[prefix + 'selected_ratio'] = count.float() / denom
                diagnostics[prefix + 'candidate_count'] = candidate.sum()
                diagnostics[prefix + 'selected_count'] = count
                diagnostics[prefix + 'updated_count'] = (
                    count * count.ge(self.min_pixels) * int(update_active))
        if not update_active:
            return self._empty_update_log(feat)
        return self._update_from_mask(feat_sel, label, valid, domain='target')

    @torch.no_grad()
    def _refresh_fused_prototypes(self):
        if self.source_prototypes.numel() == 0:
            return
        alpha = float(self.prototype_fusion_weight)
        for cls in range(self.num_classes):
            s_valid = self.source_valid_proto[cls]
            t_valid = self.target_valid_proto[cls]
            if s_valid and t_valid:
                proto = alpha * self.source_prototypes[cls] + (
                    1.0 - alpha) * self.target_prototypes[cls]
            elif s_valid:
                proto = self.source_prototypes[cls]
            elif t_valid:
                proto = self.target_prototypes[cls]
            else:
                continue
            self.prototypes[cls] = F.normalize(proto, dim=0)
            self.valid_proto[cls] = True

    def _similarity(self, feat, read_only=False):
        feat = self._select_feat(feat)
        if self.separate_domain_prototypes and not read_only:
            self._refresh_fused_prototypes()
        feat_norm = F.normalize(feat, dim=1)
        proto_norm = F.normalize(self.prototypes.detach(), dim=1)
        sim = torch.einsum('bchw,kc->bkhw', feat_norm, proto_norm)
        invalid = ~self.valid_proto
        if invalid.any():
            sim = sim.masked_fill(invalid.view(1, -1, 1, 1), -2.0)
        return sim

    def _weight_debug(self, weight, margin=None, label=None):
        # proto_weight is also consumed by optional boundary reliability gates.
        debug = dict(proto_weight=weight.detach())
        if self.collect_debug:
            debug['proto_margin'] = (margin.detach() if margin is not None
                                     else weight.new_zeros(weight.shape))
        if not self.log_stats:
            return debug
        debug['proto_weight_mean'] = weight.detach().mean()
        debug['proto_valid_classes'] = self.valid_proto.float().sum().detach()
        valid = label.ne(self.ignore_index) if label is not None else None
        debug['proto_margin_mean'] = (
            margin.detach()[valid].mean()
            if margin is not None and valid is not None and valid.any()
            else weight.new_tensor(0.0))
        if margin is not None and self.log_class_stats:
            for cls in range(self.num_classes):
                mask = label.eq(cls)
                debug[f'proto_weight_class_{cls}_mean'] = (
                    weight.detach()[mask].mean()
                    if mask.any() else weight.new_tensor(1.0))
                debug[f'proto_margin_class_{cls}_mean'] = (
                    margin.detach()[mask].mean()
                    if mask.any() else margin.new_tensor(0.0))
        return debug

    def compute_weight(self, feat, pseudo_label, out_size=None, iter=None, read_only=False):
        feat = self._select_feat(feat)
        size = out_size if out_size is not None else pseudo_label.shape[-2:]
        if iter is not None and iter < self.weight_start:
            weight = feat.new_ones((feat.shape[0], *size))
            debug = self._weight_debug(weight)
            return weight, debug
        if self.prototypes.numel() == 0 or not self.valid_proto.any():
            weight = feat.new_ones((feat.shape[0], *size))
            debug = self._weight_debug(weight)
            return weight, debug
        if self.min_valid_classes_for_weight > 0 and \
                self.valid_proto.float().sum() < self.min_valid_classes_for_weight:
            weight = feat.new_ones((feat.shape[0], *size))
            debug = self._weight_debug(weight)
            return weight, debug

        label = self._resize_label(pseudo_label, feat.shape[-2:])
        sim = self._similarity(feat, read_only=read_only)
        safe_label = label.clamp(0, self.num_classes - 1)
        class_sim = sim.gather(1, safe_label.unsqueeze(1)).squeeze(1)
        if self.valid_proto.sum() >= 2:
            top2 = sim.topk(2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.ones_like(class_sim)
        sim01 = (class_sim + 1.0) * 0.5
        proto_weight = self.proto_min_weight + (
            1.0 - self.proto_min_weight) * sim01
        proto_weight = proto_weight.clamp(self.proto_min_weight, 1.0)
        low_margin = margin.lt(self.proto_margin_thr)
        proto_weight = torch.where(
            low_margin,
            torch.minimum(
                proto_weight,
                proto_weight.new_full(proto_weight.shape,
                                      self.proto_min_weight)),
            proto_weight)
        proto_weight = proto_weight.pow(self.proto_power)

        apply_mask = torch.zeros_like(label, dtype=torch.bool)
        for cls in self.apply_to_classes:
            apply_mask |= label.eq(cls)
        valid_label = label.ne(self.ignore_index)
        class_has_proto = self.valid_proto[safe_label]
        proto_weight = torch.where(
            apply_mask & valid_label & class_has_proto, proto_weight,
            torch.ones_like(proto_weight))

        debug = self._weight_debug(proto_weight, margin, label)

        if out_size is not None and proto_weight.shape[-2:] != out_size:
            proto_weight = F.interpolate(
                proto_weight.unsqueeze(1), size=out_size, mode='bilinear',
                align_corners=False).squeeze(1).clamp(0.0, 1.0)
            debug['proto_weight'] = proto_weight.detach()
            if 'proto_margin' in debug:
                debug['proto_margin'] = F.interpolate(
                    margin.unsqueeze(1), size=out_size, mode='bilinear',
                    align_corners=False).squeeze(1).detach()
        return proto_weight, debug

    def loss_target(self, feat, pseudo_label, final_weight, iter=None):
        feat = self._select_feat(feat)
        if iter is None or iter < self.loss_start or \
                self.prototypes.numel() == 0 or not self.valid_proto.any():
            return feat.sum() * 0.0, dict(loss_proto_tgt=feat.new_tensor(0.0))
        if self.min_valid_classes_for_loss > 0 and \
                self.valid_proto.float().sum() < self.min_valid_classes_for_loss:
            return feat.sum() * 0.0, dict(loss_proto_tgt=feat.new_tensor(0.0))

        sim_logits = self._similarity(feat) / self.proto_temp
        label = self._resize_label(pseudo_label, sim_logits.shape[-2:])
        weight = self._resize_map(final_weight, sim_logits.shape[-2:])
        safe_label = label.clamp(0, self.num_classes - 1)
        valid = label.ne(self.ignore_index) & weight.gt(0) & \
            self.valid_proto[safe_label]
        if not valid.any():
            return feat.sum() * 0.0, dict(loss_proto_tgt=feat.new_tensor(0.0))
        loss = F.cross_entropy(
            sim_logits, safe_label, reduction='none')
        loss = (loss * weight)[valid].mean()
        return loss, dict(loss_proto_tgt=loss.detach())
