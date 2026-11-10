# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Add ddp_wrapper from mmgen

import random
import os.path as osp
import re
import warnings

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, Fp16OptimizerHook, OptimizerHook,
                         build_optimizer, build_runner, wrap_fp16_model)
from mmcv.utils import build_from_cfg

from mmseg.core import DistEvalHook, EvalHook
from mmseg.core.ddp_wrapper import DistributedDataParallelWrapper
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
from mmseg.utils.checkpoint_policy import DorisCheckpointHook


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)
    from tools.doris_checkpoint_plan import configure_checkpoint_policy, check_hphy_training_checkpoint
    check_hphy_training_checkpoint(cfg)
    doris_checkpoints = configure_checkpoint_policy(cfg)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        use_ddp_wrapper = cfg.get('use_ddp_wrapper', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if use_ddp_wrapper:
            mmcv.print_log('Use DDP Wrapper.', 'mmseg')
            model = DistributedDataParallelWrapper(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    optimizer_config = cfg.optimizer_config
    optimizer_config_kwargs = {} if optimizer_config is None \
        else dict(optimizer_config)
    if fp16_cfg is not None and 'uda' not in cfg:
        optimizer_config = Fp16OptimizerHook(
            **optimizer_config_kwargs, **fp16_cfg, distributed=distributed)
    elif distributed and isinstance(optimizer_config, dict) and \
            'type' not in optimizer_config:
        optimizer_config = OptimizerHook(**optimizer_config_kwargs)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   None if doris_checkpoints else cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if doris_checkpoints:
        # Save after validation so either Best record is included in latest.
        runner.register_hook(build_from_cfg(cfg.checkpoint_config, HOOKS), priority='LOW')
    if cfg.get('custom_hooks', None):
        for hook_cfg in cfg.custom_hooks:
            runner.register_hook(build_from_cfg(hook_cfg, HOOKS))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume_from = cfg.resume_from
        runner.resume(cfg.resume_from)
        if cfg.get('uda', {}).get('type') == 'DACS':
            if cfg.get('runner', {}).get('type') == 'IterBasedRunner':
                checkpoint_name = osp.basename(osp.realpath(cfg.resume_from))
                checkpoint_iter = re.fullmatch(r'(?:best_.+_)?iter_(\d+)\.pth', checkpoint_name)
                completed = (int(checkpoint_iter.group(1)) if checkpoint_iter else
                             runner.meta.get('doris_checkpoint', {}).get('completed_iter'))
                if completed == runner.iter + 1:
                    # MMCV 1.4 names the saved checkpoint after the completed
                    # step but stores its zero-based counter in meta['iter'].
                    runner._iter += 1
                    runner._inner_iter = runner.iter
                    logger.info('Corrected MMCV resume counter to %d completed iterations', runner.iter)
            resumed_model = runner.model.module if hasattr(runner.model, 'module') else runner.model
            resumed_model.local_iter = runner.iter
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    if cfg.get('uda', {}).get('mechanism_logging', False):
        # Resume may restore historical checkpoint meta; observations must use
        # the effective configuration of this invocation, including CLI paths.
        runner.mechanism_config = cfg
        if cfg.uda.get('mechanism_online_plot', False):
            from mmseg.utils.precision_logger import MechanismSnapshotHook
            runner.register_hook(MechanismSnapshotHook(), priority='ABOVE_NORMAL')
            runner.mechanism_snapshot_hook = True
    runner.run(data_loaders, cfg.workflow)
