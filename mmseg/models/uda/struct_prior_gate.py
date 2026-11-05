import torch
import torch.nn as nn
import torch.nn.functional as F


class StructPriorConsistencyGate(nn.Module):
    """Spatial-structure prior gate for target pseudo-label weights."""

    def __init__(self,
                 num_classes,
                 struct_positive_classes=(2, 3, 4),
                 apply_to_classes=None,
                 struct_min_weight=0.6,
                 struct_filter_threshold=0.35,
                 struct_filter_mode='soft',
                 struct_warmup_iters=6000,
                 struct_rampup_iters=4000,
                 source_adj_momentum=0.99,
                 semantic_proto_momentum=0.99,
                 use_source_adjacency_ema=True,
                 use_semantic_proto=True,
                 use_target_gate=True,
                 kernel_size=3,
                 confidence_power=1.0,
                 continuity_power=1.0,
                 adjacency_power=1.0,
                 semantic_power=0.5,
                 detach_gate=True,
                 ignore_index=255,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        assert kernel_size == 3, 'Struct gate currently supports 3x3 kernels.'
        assert struct_filter_mode in ('soft', 'hard')
        self.num_classes = num_classes
        self.struct_positive_classes = list(
            struct_positive_classes if apply_to_classes is None
            else apply_to_classes)
        self.struct_min_weight = struct_min_weight
        self.struct_filter_threshold = struct_filter_threshold
        self.struct_filter_mode = struct_filter_mode
        self.struct_warmup_iters = struct_warmup_iters
        self.struct_rampup_iters = struct_rampup_iters
        self.source_adj_momentum = source_adj_momentum
        self.semantic_proto_momentum = semantic_proto_momentum
        self.use_source_adjacency_ema = use_source_adjacency_ema
        self.use_semantic_proto = use_semantic_proto
        self.use_target_gate = use_target_gate
        self.confidence_power = confidence_power
        self.continuity_power = continuity_power
        self.adjacency_power = adjacency_power
        self.semantic_power = semantic_power
        self.detach_gate = detach_gate
        self.ignore_index = ignore_index
        self.eps = eps

        kernel = torch.ones(1, 1, 3, 3)
        kernel[:, :, 1, 1] = 0
        self.register_buffer('neighbor_kernel', kernel)
        self.register_buffer('source_adj_ema',
                             torch.zeros(num_classes, num_classes))
        self.register_buffer('semantic_proto_ema',
                             torch.zeros(num_classes, num_classes))
        self.register_buffer('source_adj_initialized',
                             torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('semantic_proto_initialized',
                             torch.zeros(num_classes, dtype=torch.bool))

    def _as_label_map(self, label):
        if label.dim() == 4 and label.size(1) == 1:
            label = label.squeeze(1)
        return label.long()

    def _valid_label(self, label):
        return (label != self.ignore_index) & (label >= 0) & (
            label < self.num_classes)

    def _one_hot(self, label):
        valid = self._valid_label(label)
        safe_label = label.clamp(0, self.num_classes - 1)
        onehot = F.one_hot(safe_label, self.num_classes).permute(0, 3, 1, 2)
        return onehot.float() * valid.unsqueeze(1).float(), valid

    def _neighbor_counts(self, onehot):
        kernel = self.neighbor_kernel.to(dtype=onehot.dtype,
                                         device=onehot.device)
        kernel = kernel.repeat(self.num_classes, 1, 1, 1)
        return F.conv2d(onehot, kernel, padding=1, groups=self.num_classes)

    def _valid_neighbor_counts(self, valid):
        kernel = self.neighbor_kernel.to(dtype=valid.dtype,
                                         device=valid.device)
        return F.conv2d(valid, kernel, padding=1)

    def compute_local_neighbor_distribution(self, label_map):
        label_map = self._as_label_map(label_map)
        onehot, valid = self._one_hot(label_map)
        neigh_counts = self._neighbor_counts(onehot)
        valid_counts = self._valid_neighbor_counts(valid.unsqueeze(1).float())
        neigh_frac = neigh_counts / valid_counts.clamp_min(1.0)
        neigh_frac = neigh_frac * valid.unsqueeze(1).float()
        return neigh_frac, neigh_counts, valid

    def compute_continuity_score(self, pseudo_label):
        pseudo_label = self._as_label_map(pseudo_label)
        neigh_frac, _, valid = self.compute_local_neighbor_distribution(
            pseudo_label)
        gather_label = pseudo_label.clamp(0, self.num_classes - 1).unsqueeze(1)
        continuity = torch.gather(neigh_frac, 1, gather_label).squeeze(1)
        return continuity * valid.float()

    @torch.no_grad()
    def update_source_adjacency(self, source_gt):
        if source_gt is None or not self.use_source_adjacency_ema:
            return self.source_adj_ema
        source_gt = self._as_label_map(source_gt)
        onehot, _ = self._one_hot(source_gt)
        neigh_counts = self._neighbor_counts(onehot)
        adj = torch.einsum('bihw,bjhw->ij', onehot, neigh_counts)
        row_sum = adj.sum(dim=1, keepdim=True)
        valid_rows = row_sum.squeeze(1) > 0
        adj = adj / row_sum.clamp_min(self.eps)
        if valid_rows.any():
            rows = valid_rows
            old = self.source_adj_ema[rows]
            initialized = self.source_adj_initialized[rows].unsqueeze(1)
            updated = torch.where(
                initialized,
                self.source_adj_momentum * old +
                (1 - self.source_adj_momentum) * adj[rows], adj[rows])
            self.source_adj_ema[rows] = updated
            self.source_adj_initialized[rows] = True
        return self.source_adj_ema

    def compute_adjacency_score(self, pseudo_label):
        pseudo_label = self._as_label_map(pseudo_label)
        neigh_frac, _, valid = self.compute_local_neighbor_distribution(
            pseudo_label)
        adj = self.source_adj_ema.to(device=neigh_frac.device,
                                     dtype=neigh_frac.dtype)
        row_ready = self.source_adj_initialized.to(device=neigh_frac.device)
        safe_label = pseudo_label.clamp(0, self.num_classes - 1)
        adj_rows = adj[safe_label.reshape(-1)].reshape(*safe_label.shape,
                                                       self.num_classes)
        adj_rows = adj_rows.permute(0, 3, 1, 2)
        score = (adj_rows * neigh_frac).sum(dim=1)
        ready = row_ready[safe_label] & valid
        return torch.where(ready, score, torch.ones_like(score))

    @torch.no_grad()
    def update_semantic_prototype(self, source_probs, source_gt):
        if (source_probs is None or source_gt is None
                or not self.use_semantic_proto):
            return self.semantic_proto_ema
        source_gt = self._as_label_map(source_gt)
        if source_probs.shape[-2:] != source_gt.shape[-2:]:
            source_probs = F.interpolate(
                source_probs,
                size=source_gt.shape[-2:],
                mode='bilinear',
                align_corners=False)
        valid = self._valid_label(source_gt)
        flat_probs = source_probs.permute(0, 2, 3, 1).reshape(
            -1, self.num_classes)
        flat_gt = source_gt.reshape(-1)
        flat_valid = valid.reshape(-1)
        for c in range(self.num_classes):
            mask = flat_valid & (flat_gt == c)
            if not mask.any():
                continue
            proto = flat_probs[mask].mean(dim=0)
            if self.semantic_proto_initialized[c]:
                self.semantic_proto_ema[c] = (
                    self.semantic_proto_momentum * self.semantic_proto_ema[c] +
                    (1 - self.semantic_proto_momentum) * proto)
            else:
                self.semantic_proto_ema[c] = proto
                self.semantic_proto_initialized[c] = True
        return self.semantic_proto_ema

    def compute_semantic_score(self, target_probs, pseudo_label):
        if not self.use_semantic_proto:
            return target_probs.new_ones(pseudo_label.shape)
        pseudo_label = self._as_label_map(pseudo_label)
        proto = self.semantic_proto_ema.to(device=target_probs.device,
                                           dtype=target_probs.dtype)
        ready_rows = self.semantic_proto_initialized.to(
            device=target_probs.device)
        safe_label = pseudo_label.clamp(0, self.num_classes - 1)
        proto_pix = proto[safe_label.reshape(-1)].reshape(
            *safe_label.shape, self.num_classes).permute(0, 3, 1, 2)
        score = F.cosine_similarity(target_probs, proto_pix, dim=1, eps=self.eps)
        score = score.clamp(0.0, 1.0)
        ready = ready_rows[safe_label] & self._valid_label(pseudo_label)
        return torch.where(ready, score, torch.ones_like(score))

    def _class_mask(self, label, classes):
        class_tensor = torch.tensor(classes, device=label.device)
        return torch.any(label[..., None] == class_tensor, dim=-1)

    def _matrix_entropy(self, matrix):
        row_sum = matrix.sum(dim=1, keepdim=True).clamp_min(self.eps)
        p = matrix / row_sum
        ent = -(p * (p + self.eps).log()).sum(dim=1)
        ready = matrix.sum(dim=1) > 0
        if ready.any():
            return ent[ready].mean()
        return matrix.new_tensor(0.0)

    def _adjacency_matrix_from_label(self, label):
        label = self._as_label_map(label)
        onehot, _ = self._one_hot(label)
        neigh_counts = self._neighbor_counts(onehot)
        adj = torch.einsum('bihw,bjhw->ij', onehot, neigh_counts)
        row_sum = adj.sum(dim=1, keepdim=True)
        return adj / row_sum.clamp_min(self.eps)

    def apply(self,
              target_logits=None,
              target_probs=None,
              pseudo_label=None,
              pseudo_weight=None,
              source_gt=None,
              source_logits=None,
              source_probs=None,
              iter=0,
              ignore_index=None,
              collect_debug=True,
              log_stats=True):
        if ignore_index is not None:
            self.ignore_index = ignore_index
        if target_probs is None:
            target_probs = torch.softmax(target_logits, dim=1)
        if source_probs is None and source_logits is not None:
            source_probs = torch.softmax(source_logits, dim=1)
        if pseudo_label is None:
            pseudo_label = target_probs.argmax(dim=1)
        pseudo_label = self._as_label_map(pseudo_label)

        with torch.no_grad():
            self.update_source_adjacency(source_gt)
            self.update_semantic_prototype(source_probs, source_gt)

        conf_t = target_probs.max(dim=1)[0]
        r_cont = self.compute_continuity_score(pseudo_label)
        r_adj = self.compute_adjacency_score(pseudo_label)
        r_sem = self.compute_semantic_score(target_probs, pseudo_label)
        r_struct = (
            conf_t.clamp(0, 1).pow(self.confidence_power) *
            r_cont.clamp(0, 1).pow(self.continuity_power) *
            r_adj.clamp(0, 1).pow(self.adjacency_power) *
            r_sem.clamp(0, 1).pow(self.semantic_power)).clamp(0, 1)
        if self.detach_gate:
            r_struct = r_struct.detach()
            r_cont = r_cont.detach()
            r_adj = r_adj.detach()
            r_sem = r_sem.detach()

        struct_mask = self._class_mask(pseudo_label,
                                       self.struct_positive_classes)
        if not self.use_target_gate:
            if not collect_debug and not log_stats:
                return pseudo_weight, dict(
                    R_struct=r_struct.detach(), R_continuity=r_cont.detach(),
                    R_adj=r_adj.detach(), R_sem=r_sem.detach(),
                    final_pseudo_weight=pseudo_weight.detach())
            target_adj = self._adjacency_matrix_from_label(pseudo_label).detach()
            debug_dict = dict(
                R_struct=r_struct.detach(),
                R_continuity=r_cont.detach(),
                R_adj=r_adj.detach(),
                R_sem=r_sem.detach(),
                source_adj_matrix=self.source_adj_ema.detach().clone(),
                target_adj_matrix=target_adj,
                struct_mask=struct_mask.detach(),
                final_pseudo_weight=pseudo_weight.detach(),
                struct_gate=torch.ones_like(pseudo_weight),
                struct_gate_mean=pseudo_weight.new_tensor(1.0),
                struct_gate_struct_class_mean=pseudo_weight.new_tensor(1.0),
                struct_retain_ratio=pseudo_weight.new_tensor(1.0),
                struct_filter_ratio=pseudo_weight.new_tensor(0.0),
                struct_continuity_mean=r_cont[struct_mask].detach().mean()
                if struct_mask.any() else r_cont.new_tensor(1.0),
                struct_adj_mean=r_adj[struct_mask].detach().mean()
                if struct_mask.any() else r_adj.new_tensor(1.0),
                struct_sem_mean=r_sem[struct_mask].detach().mean()
                if struct_mask.any() else r_sem.new_tensor(1.0),
                source_adj_entropy=self._matrix_entropy(self.source_adj_ema),
                target_struct_pixel_ratio=struct_mask.float().mean())
            for cls in self.struct_positive_classes:
                key = {2: 'building', 3: 'water', 4: 'road'}.get(cls, str(cls))
                debug_dict[f'struct_gate_{key}_mean'] = (
                    pseudo_weight.new_tensor(1.0))
            return pseudo_weight, debug_dict
        gate = torch.ones_like(pseudo_weight)
        struct_gate = r_struct.clamp_min(self.struct_min_weight)
        unreliable = struct_mask & r_struct.lt(self.struct_filter_threshold)
        if self.struct_filter_mode == 'hard':
            struct_gate = torch.where(unreliable, torch.zeros_like(struct_gate),
                                      struct_gate)
        else:
            struct_gate = torch.where(unreliable, struct_gate * 0.5,
                                      struct_gate)
        gate[struct_mask] = struct_gate[struct_mask]

        if iter < self.struct_warmup_iters:
            final_gate = torch.ones_like(gate)
        else:
            ramp_den = max(float(self.struct_rampup_iters), 1.0)
            ramp = min(1.0, max(0.0, (iter - self.struct_warmup_iters) /
                                ramp_den))
            final_gate = (1 - ramp) * torch.ones_like(gate) + ramp * gate
        new_pseudo_weight = pseudo_weight * final_gate

        if not collect_debug and not log_stats:
            return new_pseudo_weight, dict(
                R_struct=r_struct.detach(), R_continuity=r_cont.detach(),
                R_adj=r_adj.detach(), R_sem=r_sem.detach(),
                final_pseudo_weight=new_pseudo_weight.detach())

        target_adj = self._adjacency_matrix_from_label(pseudo_label).detach()
        source_adj = self.source_adj_ema.detach().clone()
        struct_values = final_gate[struct_mask]
        filtered = unreliable & struct_mask
        debug_dict = dict(
            R_struct=r_struct.detach(),
            R_continuity=r_cont.detach(),
            R_adj=r_adj.detach(),
            R_sem=r_sem.detach(),
            source_adj_matrix=source_adj,
            target_adj_matrix=target_adj,
            struct_mask=struct_mask.detach(),
            final_pseudo_weight=new_pseudo_weight.detach(),
            struct_gate=final_gate.detach(),
            struct_gate_mean=final_gate.detach().mean(),
            struct_gate_struct_class_mean=struct_values.detach().mean()
            if struct_values.numel() > 0 else final_gate.new_tensor(1.0),
            struct_retain_ratio=(struct_values > 0).float().mean()
            if struct_values.numel() > 0 else final_gate.new_tensor(1.0),
            struct_filter_ratio=filtered.float().sum() /
            struct_mask.float().sum().clamp_min(1.0),
            struct_continuity_mean=r_cont[struct_mask].detach().mean()
            if struct_mask.any() else r_cont.new_tensor(1.0),
            struct_adj_mean=r_adj[struct_mask].detach().mean()
            if struct_mask.any() else r_adj.new_tensor(1.0),
            struct_sem_mean=r_sem[struct_mask].detach().mean()
            if struct_mask.any() else r_sem.new_tensor(1.0),
            source_adj_entropy=self._matrix_entropy(source_adj),
            target_struct_pixel_ratio=struct_mask.float().mean())
        for cls in self.struct_positive_classes:
            cls_mask = pseudo_label == cls
            key = {2: 'building', 3: 'water', 4: 'road'}.get(cls, str(cls))
            debug_dict[f'struct_gate_{key}_mean'] = (
                final_gate[cls_mask].detach().mean()
                if cls_mask.any() else final_gate.new_tensor(1.0))
        return new_pseudo_weight, debug_dict
