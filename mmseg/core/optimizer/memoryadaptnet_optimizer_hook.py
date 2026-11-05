import os

import torch
from mmcv.runner import HOOKS, Hook


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class MemoryAdaptNetOptimizerStateHook(Hook):
    """Save and restore MemoryAdaptNet discriminator optimizer states."""

    def __init__(self, interval=1000):
        self.interval = interval

    def before_run(self, runner):
        model = _unwrap_model(runner.model)
        if not hasattr(model, 'load_memoryadaptnet_optimizer_state_dict'):
            return
        resume_from = getattr(runner, 'resume_from', None)
        if not resume_from:
            return
        sidecar = resume_from + '.memoryadaptnet_optimizers.pth'
        if os.path.exists(sidecar):
            state = torch.load(sidecar, map_location='cpu')
            model.load_memoryadaptnet_optimizer_state_dict(state)

    def after_train_iter(self, runner):
        if (runner.iter + 1) % self.interval != 0:
            return
        model = _unwrap_model(runner.model)
        if not hasattr(model, 'memoryadaptnet_optimizer_state_dict'):
            return
        sidecar = os.path.join(
            runner.work_dir,
            f'iter_{runner.iter + 1}.pth.memoryadaptnet_optimizers.pth')
        torch.save(model.memoryadaptnet_optimizer_state_dict(), sidecar)
