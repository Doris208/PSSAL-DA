"""Checkpoint management only; model forward/update code is intentionally untouched."""
from pathlib import Path
import re
import shutil

import torch
from mmcv.runner import HOOKS, CheckpointHook
from mmcv.runner.dist_utils import master_only


def checkpoint_bytes(runner):
    def size(value):
        if torch.is_tensor(value):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(size(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(size(v) for v in value)
        return 0
    total = size(runner.model.state_dict())
    optimizers = runner.optimizer if isinstance(runner.optimizer, dict) else {'optimizer': runner.optimizer}
    return total + sum(size(o.state_dict()) for o in optimizers.values() if o is not None)


def prepare_save(runner, path):
    """Never overwrite an existing numbered/best checkpoint or silently prune."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('Refusing to overwrite checkpoint: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    needed = int(checkpoint_bytes(runner) * 1.1) + 64 * 1024**2
    free = shutil.disk_usage(path.parent).free
    if free < needed:
        raise OSError('Insufficient free space for checkpoint: need approximately {} bytes, '
                      'available {}; no history was deleted'.format(needed, free))
    runner.meta.setdefault('doris_checkpoint', {})['completed_iter'] = runner.iter + 1


@HOOKS.register_module()
class DorisCheckpointHook(CheckpointHook):
    """One saver for the union of periodic and config-derived transition nodes."""

    def __init__(self, key_iterations=(), **kwargs):
        super().__init__(**kwargs)
        if self.by_epoch or self.interval != 2000 or self.max_keep_ckpts != -1 or not self.save_optimizer:
            raise ValueError('DORIS requires iteration checkpoints, interval=2000, unlimited retention and optimizer state')
        if self.out_dir is not None or self.args.get('filename_tmpl', 'iter_{}.pth') != 'iter_{}.pth':
            raise ValueError('DORIS checkpoints must use the experiment work_dir and iter_{}.pth names')
        self.key_iterations = set(key_iterations)

    def before_run(self, runner):
        super().before_run(runner)
        if sum(isinstance(h, CheckpointHook) for h in runner.hooks) != 1:
            raise ValueError('Expected exactly one checkpoint saving Hook')
        if runner.meta is None:
            runner.meta = {}
        if runner.rank == 0:
            existing = [re.fullmatch(r'(?:best_.+_)?iter_(\d+)\.pth', p.name)
                        for p in Path(self.out_dir).glob('*.pth')]
            if any(int(m[1]) > runner.iter for m in existing if m):
                raise FileExistsError('Work directory contains checkpoints beyond the resume Iter; '
                                      'continue in a new --work-dir')
            remaining = {n for n in range(2000, runner.max_iters + 1, 2000)} | self.key_iterations | {runner.max_iters}
            remaining = {n for n in remaining if n > runner.iter}
            runner.logger.info('DORIS keeps all periodic/key checkpoints: %s. Current-state lower-bound '
                'estimate including two Best files: %.2f GiB; optimizer state may still grow.',
                sorted(remaining), checkpoint_bytes(runner) * (len(remaining) + 2) / 1024**3)

    def every_n_iters(self, runner, n):
        return runner.iter + 1 in self.key_iterations or super().every_n_iters(runner, n)

    @master_only
    def _save_checkpoint(self, runner):
        path = Path(self.out_dir) / 'iter_{}.pth'.format(runner.iter + 1)
        prepare_save(runner, path)
        # Set before serialization, so latest also contains the current pointer.
        runner.meta.setdefault('hook_msgs', {})['last_ckpt'] = str(path)
        super()._save_checkpoint(runner)
