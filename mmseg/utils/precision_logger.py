# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Increase float output precision

# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os
import os.path as osp
import time
from collections import OrderedDict

import mmcv
import torch
import torch.distributed as dist
from mmcv.fileio.file_client import FileClient
from mmcv.runner import HOOKS, Hook, LoggerHook
from mmcv.utils import is_tuple_of, scandir


@HOOKS.register_module()
class PrecisionLoggerHook(LoggerHook):
    """Logger hook in text.

    In this logger hook, the information will be printed on terminal and
    saved in json file.

    Args:
        by_epoch (bool, optional): Whether EpochBasedRunner is used.
            Default: True.
        interval (int, optional): Logging interval (every k iterations).
            Default: 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
            epoch if less than :attr:`interval`. Default: True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Default: False.
        interval_exp_name (int, optional): Logging interval for experiment
            name. This feature is to help users conveniently get the experiment
            information from screen or log file. Default: 1000.
        out_dir (str, optional): Logs are saved in ``runner.work_dir`` default.
            If ``out_dir`` is specified, logs will be copied to a new directory
            which is the concatenation of ``out_dir`` and the last level
            directory of ``runner.work_dir``. Default: None.
            `New in version 1.3.16.`
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be copied to ``out_dir``.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.3.16.`
        keep_local (bool, optional): Whether to keep local log when
            :attr:`out_dir` is specified. If False, the local log will be
            removed. Default: True.
            `New in version 1.3.16.`
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`
    """

    def __init__(self,
                 by_epoch=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 interval_exp_name=1000,
                 out_dir=None,
                 out_suffix=('.log.json', '.log', '.py'),
                 keep_local=True,
                 console_keys=None,
                 file_client_args=None,
                 json_keys=None,
                 output_mode='minimal',
                 paper_process_logging=False,
                 log_memory=False,
                 paper_names=False,
                 process_interval=1000):
        super(PrecisionLoggerHook, self).__init__(interval, ignore_last,
                                                  reset_flag, by_epoch)
        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.interval_exp_name = interval_exp_name
        self.console_keys = None if console_keys is None else set(console_keys)
        self.json_keys = None if json_keys is None else set(json_keys)
        if output_mode not in ('minimal', 'diagnostic'):
            raise ValueError('output_mode must be minimal or diagnostic')
        self.output_mode = output_mode
        self.paper_process_logging = paper_process_logging
        self.log_memory = log_memory
        self.paper_names = paper_names
        self.process_interval = process_interval

        if out_dir is None and file_client_args is not None:
            raise ValueError(
                'file_client_args should be "None" when `out_dir` is not'
                'specified.')
        self.out_dir = out_dir

        if not (out_dir is None or isinstance(out_dir, str)
                or is_tuple_of(out_dir, str)):
            raise TypeError('out_dir should be  "None" or string or tuple of '
                            'string, but got {out_dir}')
        self.out_suffix = out_suffix

        self.keep_local = keep_local
        self.file_client_args = file_client_args
        if self.out_dir is not None:
            self.file_client = FileClient.infer_client(file_client_args,
                                                       self.out_dir)

    def before_run(self, runner):
        super(PrecisionLoggerHook, self).before_run(runner)

        if self.out_dir is not None:
            self.file_client = FileClient.infer_client(self.file_client_args,
                                                       self.out_dir)
            # The final `self.out_dir` is the concatenation of `self.out_dir`
            # and the last level directory of `runner.work_dir`
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)
            runner.logger.info(
                (f'Text logs will be saved to {self.out_dir} by '
                 f'{self.file_client.name} after the training process.'))

        self.start_iter = runner.iter
        self.mechanism = None
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if runner.rank == 0 and getattr(model, 'mechanism_logging', False):
            from .mechanism import MechanismCollector
            self.mechanism = MechanismCollector(runner, model)
        self.json_log_path = osp.join(runner.work_dir,
                                      f'{runner.timestamp}.log.json')
        if runner.meta is not None:
            self._dump_log(runner.meta, runner)

    def _get_max_memory(self, runner):
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _keep_train_key(self, name, keys):
        if name in ('phy_gate_iter', 'phy_gate_rho', 'phy_gate_active_ratio',
                    'phy_S_mean_active', 'phy_A_mean_active', 'phy_A_max_active'):
            return True
        if name == 'train_iter_time':
            return True
        if getattr(self, 'paper_names', False) and name.startswith(
                ('mparc.', 'pbcr.', 'teacher.', 'schedule.')):
            return True
        if name in ('mode', 'epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory') or 'loss' in name or name.split('.')[-1] == 'acc_seg':
            return True
        if name.startswith(('doris.', 'hphy_')):
            # These scalars are emitted only by the explicit model-side opt-in.
            return True
        if self.paper_process_logging and name in (
                'teacher_target_confidence_mean', 'mparc_pseudo_weight_mean',
                'proto_tgt_update_ratio', 'struct_edge_w_mix_mean'):
            return True
        if self.output_mode == 'diagnostic':
            return keys is None or name in keys
        return keys is not None and name in keys

    def _paper_log_values(self, values, runner):
        """Presentation only: keep training loss keys out of this renaming step."""
        result = OrderedDict()
        snapshot = (runner.iter + 1) % self.process_interval == 0
        names = dict(loss='loss_total', loss_phy_src='mparc.loss_phy_source_raw',
                     loss_phy_tgt='mparc.loss_phy_target',
                     loss_proto_tgt='pbcr.loss_proto_target',
                     **{'doris.phy_source_weighted': 'mparc.loss_phy_source'})
        for key, value in values.items():
            sampled = (key.startswith('doris.') and key != 'doris.phy_source_weighted') or \
                '.edge_diag.' in key or '.edge_count.' in key or '.edge_quality.' in key
            if sampled:
                if not snapshot:
                    continue
                # MMCV averages sparse histories by sample count, not iteration.
                # Use the current sampled scalar; never repeat or average old snapshots.
                history = getattr(runner.log_buffer, 'val_history', {}).get(key, [])
                if history:
                    value = history[-1]
            if key.startswith('doris.'):
                if key not in names and key.split('.')[1] not in (
                        'mparc', 'pbcr', 'teacher', 'schedule'):
                    # Compatibility fields remain model-side; the paper view uses clear scopes.
                    continue
                name = names.get(key, key[len('doris.'):])
            elif '.edge_diag.' in key or '.edge_count.' in key or '.edge_quality.' in key:
                branch = 'mix' if key.startswith('mix.') else 'source'
                scale = 'hr.' if '.hr.' in key else ''
                leaf = key.rsplit('.', 1)[-1]
                if '.edge_quality.' in key:
                    leaf = ('local_boundary_' + leaf +
                            ('_pseudo_agreement' if branch == 'mix' else ''))
                elif '.edge_diag.' in key:
                    leaf = 'loss_edge_' + leaf
                name = 'pbcr.' + branch + '.' + scale + leaf
            else:
                name = names.get(key, key)
                name = name.replace('mix.decode.', 'mix.').replace('decode.', 'source.')
                name = name.replace('loss_struct_edge', 'loss_edge')
                if name.startswith('mix.'):
                    name = name.replace('loss_seg', 'loss_pseudo')
            result[name] = value
        return result

    def _log_info(self, log_dict, runner):
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'

            # by epoch: Epoch [4][100/1000]
            # by iter:  Iter [100/100000]
            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}]' \
                          f'[{log_dict["iter"]}/{len(runner.data_loader)}]\t'
            else:
                log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'
            log_str += f'{lr_str}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, '
                if 'data_time' in log_dict:
                    log_str += f'data_time: {log_dict["data_time"]:.3f}, '
                # statistic memory
                if 'memory' in log_dict:
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) ' \
                    f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        minimal_train = log_dict['mode'] == 'train' and self.output_mode == 'minimal'
        log_items = []
        detail_items = OrderedDict(
            source=[], mix=[], target=[], aux=[], stats=[])
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if log_dict['mode'] == 'train' and not self._keep_train_key(
                    name, self.console_keys):
                continue
            is_train_acc = (log_dict['mode'] == 'train' and
                            name.split('.')[-1] == 'acc_seg')
            if is_train_acc:
                # Existing MMSeg accuracy is already on a 0..100 scale.
                val = f'{val:.2f}%'
            elif isinstance(val, float):
                val = (f'{val:.3e}' if 'loss' in name and 0 < abs(val) < 1e-4
                       else f'{val:.4f}')
            item = f'{name}: {val}'
            if minimal_train and name in ('loss', 'loss_total'):
                log_items.insert(0, item)
            elif not minimal_train or is_train_acc:
                log_items.append(item)
            elif getattr(self, 'paper_names', False) and name.startswith(
                    ('mparc.', 'pbcr.', 'teacher.', 'schedule.')):
                group = ('.'.join(name.split('.')[:3]) if name.startswith('pbcr.proto.')
                         else name.split('.')[0])
                detail_items.setdefault(group, []).append(item)
            elif 'loss' in name:
                if name.startswith(('decode.', 'source.')):
                    detail_items['source'].append(item)
                elif name.startswith('mix.'):
                    detail_items['mix'].append(item)
                elif name.startswith('tgt.'):
                    detail_items['target'].append(item)
                else:
                    detail_items['aux'].append(item)
            else:
                detail_items['stats'].append(item)
        log_str += ', '.join(log_items)
        if minimal_train:
            for group, items in detail_items.items():
                if items:
                    log_str += f'\n    {group}: ' + ', '.join(items)

        runner.logger.info(log_str)

    def _dump_log(self, log_dict, runner):
        # dump log in json format
        json_log = OrderedDict()
        for k, v in log_dict.items():
            if log_dict.get('mode') == 'train' and not self._keep_train_key(
                    k, self.json_keys):
                continue
            json_log[k] = self._round_float(v)
        # only append log at last line
        if runner.rank == 0:
            with open(self.json_log_path, 'a+') as f:
                mmcv.dump(json_log, f, file_format='json')
                f.write('\n')

    def _round_float(self, items):
        return items

    def log(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
            if not self.by_epoch:
                runner.log_buffer.output['eval_batches'] = cur_iter
                cur_iter = runner.iter + 1
            # IterTimerHook can add `time` after evaluation cleared the buffer.
            # The explicit evaluation marker takes precedence over that timer.
            mode = 'val'
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)
            mode = self.get_mode(runner)

        log_dict = OrderedDict(
            mode=mode,
            epoch=self.get_epoch(runner),
            iter=cur_iter)

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        if self.log_memory and mode == 'train' and \
                'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)
        if mode == 'train' and getattr(self, 'paper_names', False):
            log_dict = self._paper_log_values(log_dict, runner)
        if mode == 'val':
            log_dict.pop('time', None)
            log_dict.pop('data_time', None)

        self._log_info(log_dict, runner)
        self._dump_log(log_dict, runner)
        return log_dict

    def after_train_iter(self, runner):
        start = time.time()
        if getattr(self, 'mechanism', None) is not None and not getattr(
                runner, 'mechanism_snapshot_hook', False):
            self.mechanism.consume()
            # IterTimer ran before this logger. Exclude observation from the next
            # iteration's data_time/time without resetting the CUDA peak statistic.
            elapsed = time.time() - start
            for hook in runner.hooks:
                if hook.__class__.__name__ == 'IterTimerHook' and hasattr(hook, 't'):
                    hook.t += elapsed
        super(PrecisionLoggerHook, self).after_train_iter(runner)

    def after_run(self, runner):
        if getattr(self, 'mechanism', None) is not None and runner.iter >= runner.max_iters:
            self.mechanism.finish()
        # copy or upload logs to self.out_dir
        if self.out_dir is not None:
            for filename in scandir(runner.work_dir, self.out_suffix, True):
                local_filepath = osp.join(runner.work_dir, filename)
                out_filepath = self.file_client.join_path(
                    self.out_dir, filename)
                with open(local_filepath, 'r') as f:
                    self.file_client.put_text(f.read(), out_filepath)

                runner.logger.info(
                    (f'The file {local_filepath} has been uploaded to '
                     f'{out_filepath}.'))

                if not self.keep_local:
                    os.remove(local_filepath)
                    runner.logger.info(
                        (f'{local_filepath} was removed due to the '
                         '`self.keep_local=False`'))


class MechanismSnapshotHook(Hook):
    """Run the existing collector before evaluation, keeping logger order unchanged."""

    def after_train_iter(self, runner):
        start = time.time()
        for hook in runner.hooks:
            collector = getattr(hook, 'mechanism', None)
            if collector is not None:
                collector.consume()
        elapsed = time.time() - start
        for hook in runner.hooks:
            if hook.__class__.__name__ == 'IterTimerHook' and hasattr(hook, 't'):
                hook.t += elapsed
