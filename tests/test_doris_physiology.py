"""Small physiology loss/collection checks, without a training dataset or GPU."""
import ast
import unittest

from test_minimal_output import ROOT, torch


@unittest.skipIf(torch is None, 'Torch unavailable')
class PhysiologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / 'mmseg/models/uda/phy_reliability.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        cls.scope = dict(torch=torch, F=torch.nn.functional)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'phy_loss', 'exec'), cls.scope)

    def test_positive_negative_confidence_ignore_spatial_masks_and_gradients(self):
        pred = torch.full((1, 1, 1, 8), .6, requires_grad=True)
        label = torch.tensor([[[0, 1, 2, 4, 255, 3, 0, 2]]])
        confidence = torch.tensor([[[1., 1., 1., 1., 1., .8, 1., .8]]])
        valid = torch.ones(1, 1, 1, 8); valid[..., 6] = 0
        loss, diag = self.scope['target_phy_loss'](pred, label, confidence, valid,
                                                  mode='balanced_high_conf')
        self.assertEqual(diag['pos_count'], 2); self.assertEqual(diag['neg_count'], 2)
        expected = -.5 * (torch.log(torch.tensor(.6)) + torch.log(torch.tensor(.4)))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue((pred.grad[..., :2] < 0).all())
        self.assertTrue((pred.grad[..., 2:4] > 0).all())
        self.assertTrue((pred.grad[..., 4:] == 0).all())

    def test_empty_groups_renormalize_and_both_empty_backpropagate_zero(self):
        for label_id, target in ((0, 1.), (3, 0.)):
            pred = torch.full((1, 1, 2, 2), .6, requires_grad=True)
            label = torch.full((1, 2, 2), label_id)
            loss, _ = self.scope['target_phy_loss'](pred, label, torch.ones_like(label).float(),
                                                   mode='balanced_high_conf')
            expected = torch.nn.functional.binary_cross_entropy(pred, torch.full_like(pred, target))
            torch.testing.assert_close(loss, expected)
            loss.backward(); self.assertTrue(torch.isfinite(pred.grad).all())
        pred = torch.full((1, 1, 2, 2), .6, requires_grad=True)
        loss, _ = self.scope['target_phy_loss'](pred, torch.zeros(1, 2, 2).long(),
            torch.zeros(1, 2, 2), mode='balanced_high_conf')
        self.assertEqual(loss.item(), 0); loss.backward()
        self.assertTrue((pred.grad == 0).all())

    def test_imbalanced_groups_have_equal_group_weight(self):
        pred = torch.full((1,1,1,8),.6,requires_grad=True)
        label = torch.tensor([[[0,2,2,2,2,2,2,2]]])
        loss, diag = self.scope['target_phy_loss'](pred,label,torch.ones_like(label).float(),
                                                   mode='balanced_high_conf')
        expected = -.5*(torch.log(torch.tensor(.6))+torch.log(torch.tensor(.4)))
        torch.testing.assert_close(loss,expected)
        self.assertEqual(diag['pos_count'],1);self.assertEqual(diag['neg_count'],7)
        loss.backward()
        self.assertAlmostEqual(pred.grad[0,0,0,0].item(),-.5/.6,places=6)
        self.assertAlmostEqual(pred.grad[0,0,0,1:].sum().item(),.5/.4,places=6)

    def test_legacy_is_bit_exact_to_original_positive_only_bce(self):
        pred = torch.sigmoid(torch.randn(2, 1, 8, 8)).detach().requires_grad_(True)
        label = torch.arange(128).reshape(2, 8, 8) % 5
        confidence = torch.rand(2, 8, 8)
        confidence[:, 0] = 1.0
        vegetation = (label == 0) | (label == 1)
        expected = self.scope['masked_bce_loss'](pred, vegetation.unsqueeze(1),
            (vegetation & confidence.ge(.968)).unsqueeze(1))
        actual, _ = self.scope['target_phy_loss'](pred, label, confidence, torch.zeros_like(label), mode='legacy')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_grad = torch.autograd.grad(expected, pred, retain_graph=True)[0]
        torch.testing.assert_close(torch.autograd.grad(actual, pred)[0], expected_grad, rtol=0, atol=0)

    def test_new_loss_is_called_once_and_backward_reaches_existing_head(self):
        from test_doris_stage1 import new_model
        model = new_model()
        model.local_iter = 99
        model.mechanism_cfg['output_mode']='detailed'
        phy_tree = ast.parse((ROOT / 'mmseg/models/uda/phy_reliability.py').read_text())
        head = next(n for n in phy_tree.body if isinstance(n, ast.ClassDef) and n.name == 'PhyReliabilityHead')
        def conv_module(in_channels, out_channels, kernel_size, padding, **kwargs):
            return torch.nn.Sequential(torch.nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding),
                torch.nn.BatchNorm2d(out_channels),torch.nn.ReLU())
        head_scope = dict(nn=torch.nn, torch=torch, ConvModule=conv_module)
        exec(compile(ast.Module(body=[head],type_ignores=[]),'real_phy_head','exec'),head_scope)
        model.phy_head = head_scope['PhyReliabilityHead'](**model.phy_cfg['phy_head'])
        # Execute the actual target-loss block from forward_train with two classes.
        tree = ast.parse((ROOT / 'mmseg/models/uda/dacs.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DACS')
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_train')
        block = next(n for n in forward.body if isinstance(n, ast.If) and
                     ast.dump(n.test) == ast.dump(ast.parse(
                         'self.enable_phy and self.lambda_phy_tgt > 0', mode='eval').body))
        model.mechanism_records = [dict(phy_losses={})]
        vi = torch.rand(1, 3, 8, 8)
        response = model.phy_head(vi)
        self.assertTrue(torch.isfinite(response).all() and response.ge(0).all() and response.le(1).all())
        label = torch.zeros(1, 8, 8).long(); label[:, :, 4:] = 2
        calls = []
        def loss(*args, **kwargs):
            calls.append(1)
            return self.scope['target_phy_loss'](*args, **kwargs)
        before = len(list(model.parameters()))
        scope = dict(self=model, pseudo_label=label, teacher_target_confidence_map=torch.ones(1,8,8),
                     valid_pseudo_mask=torch.ones(1,1,8,8), r_phy_tgt=response,
                     target_phy_loss=loss, log_vars={})
        exec(compile(ast.Module(body=[block], type_ignores=[]), 'actual_phy_train_block', 'exec'), scope)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(list(model.parameters())), before)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.phy_head.parameters()))
        self.assertEqual(set(scope['log_vars']), {'loss_phy_tgt'})
        self.assertGreater(model.mechanism_records[0]['phy_losses']['neg_count'], 0)


if __name__ == '__main__':
    unittest.main()
