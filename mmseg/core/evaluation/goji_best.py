"""An independent Goji best record using the existing validation results once."""
import json
import math
from numbers import Real
from pathlib import Path
import re

from mmseg.utils.checkpoint_policy import prepare_save


def valid_iou(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


class GojiBestMixin:
    def __init__(self, *args, save_best_goji=False, **kwargs):
        self.save_best_goji = save_best_goji
        super().__init__(*args, **kwargs)
        if save_best_goji and (self.by_epoch or self.save_best != 'mIoU'):
            raise ValueError('Goji best requires the existing iteration-based Best mIoU validation')

    def before_run(self, runner):
        super().before_run(runner)
        if self.save_best_goji:
            if tuple(self.dataloader.dataset.CLASSES) != ('goji', 'vegetation', 'building', 'water', 'road'):
                raise ValueError('Expected the unchanged five-class Goji validation dataset')
            if Path(self.out_dir).resolve() != Path(runner.work_dir).resolve():
                raise ValueError('Best checkpoints must remain in this experiment work_dir')
            # Protect files that predate this invocation, including resumed bests.
            self._protected_bests = {str(p.resolve()) for p in Path(self.out_dir).glob('best_*.pth')}
            self._protected_bests.update(str(Path(p).resolve()) for p in
                (self.best_ckpt_path, runner.meta['hook_msgs'].get('best_goji_ckpt')) if p)
            messages = runner.meta['hook_msgs']
            if 'best_goji_score' not in messages:
                messages['goji_best_history_start_iter'] = runner.iter
                runner.logger.info('Independent Goji best history starts at resumed Iter %d; '
                                   'no earlier Goji best checkpoint is fabricated.', runner.iter)

    def _save_ckpt(self, runner, key_score):
        if not self.save_best_goji:
            return super()._save_ckpt(runner, key_score)
        messages = runner.meta['hook_msgs']
        step = runner.iter + 1
        goji = runner.log_buffer.output.get('IoU.goji')
        old_goji = messages.get('best_goji_score')
        goji_improved = valid_iou(goji) and (not valid_iou(old_goji) or goji > old_goji)
        miou_improved = valid_iou(key_score) and (
            not valid_iou(messages.get('best_score')) or key_score > messages['best_score'])
        goji_path = Path(self.out_dir) / 'best_goji_mIoU_iter_{}.pth'.format(step)
        old_goji_path = messages.get('best_goji_ckpt')
        old_miou_path = self.best_ckpt_path
        if goji_improved:
            prepare_save(runner, goji_path)
        if miou_improved:
            prepare_save(runner, Path(self.out_dir) / 'best_mIoU_iter_{}.pth'.format(step))
        # Stage both records before either serialization, for resume from either best.
        if goji_improved:
            messages.update(best_goji_score=float(goji), best_goji_ckpt=str(goji_path), best_goji_iter=step)
        if miou_improved:
            messages['best_miou_iter'] = step
            # The original comparison cannot recover from a legacy NaN best.
            if not valid_iou(messages.get('best_score')):
                messages.pop('best_score', None)
            # Defer deletion until new saves succeed; preserve pre-run history.
            self.best_ckpt_path = None
            # Keep MMCV's actual mIoU comparison, naming and save implementation.
            super()._save_ckpt(runner, key_score)
        if goji_improved:
            runner.save_checkpoint(self.out_dir, goji_path.name, create_symlink=False, save_optimizer=True)
            runner.logger.info('Best IoU.goji %.6f at Iter %d: %s', goji, step, goji_path)
        for improved, old_path in ((miou_improved, old_miou_path), (goji_improved, old_goji_path)):
            if improved and old_path and str(Path(old_path).resolve()) not in self._protected_bests:
                old = Path(old_path)
                if old.parent.resolve() == Path(self.out_dir).resolve() and old.is_file():
                    old.unlink()
        if not valid_iou(goji):
            runner.logger.warning('Missing/invalid validation IoU.goji; independent best is unchanged')
        self._write_best_report(runner)

    def _write_best_report(self, runner):
        messages = runner.meta['hook_msgs']
        records = {}
        for name, score, path in (('mIoU', 'best_score', 'best_ckpt'),
                                  ('IoU.goji', 'best_goji_score', 'best_goji_ckpt')):
            filename = messages.get(path)
            match = re.search(r'iter_(\d+)\.pth$', filename or '')
            records[name] = dict(score=float(messages[score]) if valid_iou(messages.get(score)) else None,
                iter=int(match[1]) if match else None, path=filename,
                checkpoint_present=bool(filename and Path(filename).is_file()))
        records['same_iter'] = (records['mIoU']['iter'] == records['IoU.goji']['iter']) \
            if all(records[k]['iter'] is not None for k in ('mIoU', 'IoU.goji')) else None
        records['metric_source'] = 'existing validation; IoU.goji is class 0, not Veg.mIoU'
        records['goji_history_start_iter'] = messages.get('goji_best_history_start_iter', 0)
        path = Path(self.out_dir) / 'checkpoint_best.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)
