"""Checkpoint management only; model forward/update code is intentionally untouched."""
from pathlib import Path
import shutil

import torch
from mmcv.runner import HOOKS, CheckpointHook
from mmcv.runner.dist_utils import master_only
from tools.doris_checkpoint_plan import FIXED_CHECKPOINT_ITERS


def checkpoint_bytes(runner, save_optimizer=True):
    def size(value):
        if torch.is_tensor(value):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(size(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(size(v) for v in value)
        return 0
    total = size(runner.model.state_dict())
    if not save_optimizer:
        return total
    optimizers = runner.optimizer if isinstance(runner.optimizer, dict) else {'optimizer': runner.optimizer}
    return total + sum(size(o.state_dict()) for o in optimizers.values() if o is not None)


def prepare_save(runner, path, save_optimizer=True):
    """Never overwrite an existing numbered/best checkpoint or silently prune."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('Refusing to overwrite checkpoint: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    needed = int(checkpoint_bytes(runner, save_optimizer) * 1.1) + 64 * 1024**2
    free = shutil.disk_usage(path.parent).free
    if free < needed:
        raise OSError('Insufficient free space for checkpoint: need approximately {} bytes, '
                      'available {}; no history was deleted'.format(needed, free))
    runner.meta.setdefault('doris_checkpoint', {})['completed_iter'] = runner.iter + 1
    model = getattr(runner.model, 'module', runner.model)
    gate = getattr(model, 'struct_gate', None)
    if getattr(gate, 'fusion_mode', None) == 'struct_distinct_v1':
        rule_iter = model.local_iter
        runner.meta['structural_mechanism'] = dict(config=dict(model.struct_cfg),
            rule_iter=rule_iter, gate_strength=gate.gate_strength(rule_iter),
            last_train_rule_iter=max(0, rule_iter-1),
            last_train_gate_strength=gate.gate_strength(max(0, rule_iter-1)))


@HOOKS.register_module()
class DorisCheckpointHook(CheckpointHook):
    """Roll latest every 2000 iterations and retain the three requested states."""

    def __init__(self, key_iterations=(), keep_iterations=FIXED_CHECKPOINT_ITERS, **kwargs):
        kwargs.setdefault('save_last', False)
        super().__init__(**kwargs)
        if (self.by_epoch or self.interval != 2000 or self.max_keep_ckpts != -1 or
                not self.save_optimizer or self.save_last or key_iterations):
            raise ValueError('DORIS latest requires interval=2000, optimizer, save_last=False and no key iterations')
        if self.out_dir is not None or set(self.args) - {'meta'}:
            raise ValueError('DORIS latest must use the experiment work_dir and atomic writer')
        self.keep_iterations = tuple(keep_iterations)
        if self.keep_iterations != FIXED_CHECKPOINT_ITERS:
            raise ValueError('DORIS fixed checkpoints must be Iter 2000, 4000 and 6000')

    def before_run(self, runner):
        super().before_run(runner)
        if sum(isinstance(h, CheckpointHook) for h in runner.hooks) != 1:
            raise ValueError('Expected exactly one checkpoint saving Hook')
        if runner.meta is None:
            runner.meta = {}
        runner.meta.update(self.args.get('meta', {}))
        from mmseg.utils.doris_checkpoint import initialize_policy, restore_rng_state
        if runner.rank == 0:
            initialize_policy(runner)
        restore_rng_state(runner)
        runner.logger.info('DORIS keeps four Best models, rolling latest, and fixed checkpoints at %s',
                           self.keep_iterations)

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            if self.sync_buffer:
                from mmcv.runner.dist_utils import allreduce_params
                allreduce_params(runner.model.buffers())
            from mmseg.utils.doris_checkpoint import collect_rng_states
            states = collect_rng_states()
            if runner.rank == 0:
                self._save_checkpoint(runner, states)

    @master_only
    def _save_checkpoint(self, runner, rng_states=None):
        from mmseg.utils.doris_checkpoint import save_latest
        save_latest(runner, rng_states, keep_snapshot=runner.iter + 1 in self.keep_iterations)
