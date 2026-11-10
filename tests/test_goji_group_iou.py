"""Check grouped IoUs without importing the Torch training stack."""
import ast
from pathlib import Path
import unittest

import numpy as np


class GroupIoUTests(unittest.TestCase):
    def evaluator(self, metrics):
        class Base:
            def evaluate(self, results, **kwargs):
                return dict(metrics)

        path = Path(__file__).resolve().parents[1] / 'mmseg/datasets/goji.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == 'GojiDataset')
        cls.body = [node for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'evaluate']
        cls.decorator_list = []
        tree.body = [cls]
        scope = dict(CustomDataset=Base, np=np, print_log=lambda *a, **kw: None)
        exec(compile(tree, str(path), 'exec'), scope)
        dataset = scope['GojiDataset']()
        dataset.CLASSES = ('goji', 'vegetation', 'building', 'water', 'road')
        return dataset

    def test_group_means_preserve_existing_metrics_and_ignore_nan(self):
        metrics = {'mIoU': 0.5, 'IoU.goji': 0.2, 'IoU.vegetation': 0.8,
                   'IoU.building': 0.3, 'IoU.water': np.nan, 'IoU.road': 0.9}
        result = self.evaluator(metrics).evaluate([])
        self.assertAlmostEqual(result['Veg.mIoU'], 0.5)
        self.assertAlmostEqual(result['NonVeg.mIoU'], 0.6)
        for key, value in metrics.items():
            np.testing.assert_equal(result[key], value)

    def test_non_iou_evaluation_is_unchanged(self):
        metrics = {'mF1': 0.7}
        self.assertEqual(self.evaluator(metrics).evaluate([], metric='mFscore'), metrics)


if __name__ == '__main__':
    unittest.main()
