"""Logger regression checks without constructing a training model."""
from collections import OrderedDict
import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import test_minimal_output as helpers


class LoggerDataTimeTests(unittest.TestCase):
    def hook_and_runner(self, output):
        cls = helpers.load_methods('mmseg/utils/precision_logger.py',
            'PrecisionLoggerHook', ['log', '_log_info', '_keep_train_key'],
            dict(OrderedDict=OrderedDict, datetime=datetime))
        hook = cls()
        hook.log_memory, hook.by_epoch = False, False
        hook.output_mode, hook.paper_process_logging = 'diagnostic', False
        hook.console_keys = None
        hook.interval, hook.start_iter, hook.time_sec_tot = 50, 0, 0
        hook.get_epoch = lambda r: 1
        hook.get_iter = lambda r, **kw: r.iter + 1
        hook.get_mode = lambda r: 'train' if 'time' in r.log_buffer.output else 'val'
        hook._dump_log = lambda *args: None
        messages = []
        runner = SimpleNamespace(iter=999, max_iters=20000, meta=None,
            current_lr=lambda: [6e-5], log_buffer=SimpleNamespace(output=output),
            logger=SimpleNamespace(info=messages.append))
        return hook, runner, messages

    def test_validation_marker_takes_precedence_over_time(self):
        hook, runner, messages = self.hook_and_runner(
            dict(eval_iter_num=314, time=20.0, mIoU=0.4))
        result = hook.log(runner)
        self.assertEqual(result['mode'], 'val')
        self.assertNotIn('time', result)
        self.assertIn('mIoU: 0.4000', messages[0])

    def test_train_time_without_data_time_prints_available_timing(self):
        hook, runner, messages = self.hook_and_runner(dict(time=0.5, loss=1.0))
        result = hook.log(runner)
        self.assertEqual(result['mode'], 'train')
        self.assertIn('time: 0.500', messages[0])
        self.assertNotIn('data_time:', messages[0])

    def test_cuda_peak_memory_mib_reaches_terminal_and_json_and_skips_without_cuda(self):
        fake_torch = SimpleNamespace(int=None, cuda=SimpleNamespace(
            is_available=lambda: True, max_memory_allocated=lambda device=None: 10522 * 2**20),
            tensor=lambda values, **kw: SimpleNamespace(item=lambda: values[0]))
        scope = dict(torch=fake_torch, OrderedDict=OrderedDict, datetime=datetime,
                     mmcv=SimpleNamespace(dump=lambda data, stream, **kw: json.dump(data, stream)))
        cls = helpers.load_methods('mmseg/utils/precision_logger.py', 'PrecisionLoggerHook',
            ['log', '_log_info', '_keep_train_key', '_get_max_memory', '_dump_log', '_round_float'], scope)
        template, runner, messages = self.hook_and_runner(dict(time=0.5, data_time=0.01, loss=1.0))
        hook = cls()
        hook.__dict__.update(template.__dict__)
        del hook._dump_log
        hook.log_memory, hook.json_keys = True, None
        runner.model, runner.world_size, runner.rank = SimpleNamespace(output_device=0), 1, 0
        with tempfile.TemporaryDirectory(dir=helpers.ROOT / 'outputs') as folder:
            hook.json_log_path = str(Path(folder) / 'fixture.log.json')
            result = hook.log(runner)
            self.assertEqual(result['memory'], 10522)
            self.assertIn('memory: 10522', messages[-1])
            first = json.loads(Path(hook.json_log_path).read_text().splitlines()[0])
            self.assertEqual(first['memory'], 10522)
            runner.log_buffer.output['decode.loss_struct_edge'] = 1e-8
            hook.log(runner)
            self.assertIn('1.000e-08', messages[-1])
            fake_torch.cuda.is_available = lambda: False
            self.assertNotIn('memory', hook.log(runner))


if __name__ == '__main__':
    unittest.main()
