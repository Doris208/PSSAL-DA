"""Record the existing complete validation result without saving any weights."""
from .eval_hooks import EvalHook
from tools.run_tau_proto_sensitivity import append_miou


class SensitivityEvalHook(EvalHook):
    def __init__(self, *args, csv_path, tau_proto, **kwargs):
        if (kwargs.get('save_best') or kwargs.get('save_best_goji') or
                kwargs.get('doris_best_only') or kwargs.get('by_epoch', False) or
                kwargs.get('efficient_test', False) or kwargs.get('interval') != 1000):
            raise ValueError('Sensitivity evaluation requires 1K validation without saving')
        if not csv_path:
            raise ValueError('Sensitivity evaluation requires csv_path')
        self.csv_path = csv_path
        self.tau_proto = tau_proto
        super().__init__(*args, **kwargs)

    def before_run(self, runner):
        super().before_run(runner)
        dataset = self.dataloader.dataset
        if (tuple(dataset.CLASSES) != ('goji', 'vegetation', 'building', 'water', 'road') or
                dataset.ignore_index != 255 or not dataset.reduce_zero_label):
            raise ValueError('Expected unchanged five-class Goji evaluation with ignore=255')
        if runner.iter != 0 or runner.max_iters != 15000 or runner.rank != 0:
            raise ValueError('Sensitivity runs must start at zero on one GPU for 15K')

    def evaluate(self, runner, results):
        # Remove a previous value so a missing metric cannot reuse stale output.
        runner.log_buffer.output.pop('mIoU', None)
        key_score = super().evaluate(runner, results)
        append_miou(self.csv_path, self.tau_proto, runner.iter + 1,
                    runner.log_buffer.output.get('mIoU'))
        return key_score
