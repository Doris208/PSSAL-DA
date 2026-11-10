"""Regression check for the real segmentor/head call interface; no Torch."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_method(path, class_name, method_name, scope=None):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {} if scope is None else scope
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, 'exec'), namespace)
    return namespace[method_name]


class HRDABoundaryInterfaceTests(unittest.TestCase):
    def test_segmentor_passes_boundary_mask_through_head_to_losses(self):
        forward_train = load_method('mmseg/models/decode_heads/hrda_head.py',
                                    'HRDAHead', 'forward_train')
        decode_train = load_method('mmseg/models/segmentors/encoder_decoder.py',
            'EncoderDecoder', '_decode_head_forward_train',
            dict(add_prefix=lambda losses, prefix: {
                prefix + '.' + key: value for key, value in losses.items()}))
        for mask in (object(), None):
            with self.subTest(with_mask=mask is not None):
                received = {}
                head = SimpleNamespace(enable_hr_crop=True, hr_crop_box=(0, 1, 0, 1))
                head.forward = lambda inputs, **kwargs: inputs
                def losses(logits, label, weight, edge_weight, **kwargs):
                    received.update(kwargs)
                    return dict(loss_seg=1.0)
                head.losses = losses
                head.reset_crop = lambda: setattr(head, 'hr_crop_box', None)
                head.forward_train = MethodType(forward_train, head)
                segmentor = SimpleNamespace(decode_head=head, train_cfg={})
                result = decode_train(segmentor, object(), [], object(),
                                      struct_edge_valid_mask=mask)
                self.assertEqual(result, {'decode.loss_seg': 1.0})
                self.assertIs(received.get('struct_edge_valid_mask'), mask)
                self.assertIsNone(head.hr_crop_box)


if __name__ == '__main__':
    unittest.main()
