import os

import torch
from mmcv.runner import HOOKS, Hook


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class AdvEntOptimizerStateHook(Hook):
    """Save and restore AdvEnt discriminator optimizer sidecar states."""

    def __init__(self, interval=1000):
        self.interval = interval

    def before_run(self, runner):
        model = _unwrap_model(runner.model)
        if not hasattr(model, 'load_advent_optimizer_state_dict'):
            return
        resume_from = getattr(runner, 'resume_from', None)
        if not resume_from:
            return
        sidecar = resume_from + '.advent_optimizers.pth'
        if os.path.exists(sidecar):
            state = torch.load(sidecar, map_location='cpu')
            model.load_advent_optimizer_state_dict(state)

    def after_train_iter(self, runner):
        if (runner.iter + 1) % self.interval != 0:
            return
        model = _unwrap_model(runner.model)
        if not hasattr(model, 'advent_optimizer_state_dict'):
            return
        sidecar = os.path.join(
            runner.work_dir,
            f'iter_{runner.iter + 1}.pth.advent_optimizers.pth')
        torch.save(model.advent_optimizer_state_dict(), sidecar)
