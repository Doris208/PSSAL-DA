"""Focused threshold, fusion and paper-log checks; no GPU or full model."""
from collections import OrderedDict
from types import SimpleNamespace
import unittest

import test_minimal_output as helpers
from test_minimal_output import torch


class PaperLoggerTests(unittest.TestCase):
    def test_validation_uses_training_iteration_not_dataset_length(self):
        cls = helpers.load_methods('mmseg/utils/precision_logger.py',
            'PrecisionLoggerHook', ['log'], dict(OrderedDict=OrderedDict))
        hook = cls()
        hook.paper_names, hook.log_memory = True, False
        hook.by_epoch = False
        hook.get_epoch = lambda runner: 1
        hook._log_info = hook._dump_log = lambda *args: None
        runner = SimpleNamespace(iter=999, current_lr=lambda: [6e-5],
            log_buffer=SimpleNamespace(output={'eval_iter_num': 314, 'mIoU': 0.456789}))
        result = hook.log(runner)
        self.assertEqual(result['iter'], 1000)
        self.assertEqual(result['eval_batches'], 314)
        self.assertEqual(result['mIoU'], 0.456789)

    def test_sparse_snapshot_names_do_not_repeat_or_average_old_values(self):
        cls = helpers.load_methods('mmseg/utils/precision_logger.py',
            'PrecisionLoggerHook', ['_paper_log_values'], dict(OrderedDict=OrderedDict))
        hook = cls()
        hook.process_interval = 1000
        values = {'loss': 1.2, 'loss_phy_src': 0.7,
                  'doris.phy_source_weighted': 0.056,
                  'decode.loss_struct_edge': 1e-8,
                  'mix.decode.hr.loss_seg': 0.2,
                  'decode.edge_diag.pos': 0.1,
                  'doris.mparc.r_struct_mean': 0.2}
        runner = SimpleNamespace(iter=999, log_buffer=SimpleNamespace(
            val_history={'doris.mparc.r_struct_mean': [0.2, 0.8]}))
        result = hook._paper_log_values(values, runner)
        self.assertEqual(result['mparc.r_struct_mean'], 0.8)
        self.assertEqual(result['loss_total'], 1.2)
        self.assertEqual(result['mparc.loss_phy_source'], 0.056)
        self.assertEqual(result['source.loss_edge'], 1e-8)
        self.assertEqual(result['mix.hr.loss_pseudo'], 0.2)
        self.assertEqual(result['pbcr.source.loss_edge_pos'], 0.1)
        runner.iter = 1049
        result = hook._paper_log_values(values, runner)
        self.assertNotIn('mparc.r_struct_mean', result)
        self.assertNotIn('pbcr.source.loss_edge_pos', result)
        self.assertIn('source.loss_edge', result)

    def test_diagnostic_loss_names_do_not_enter_total_twice(self):
        cls = helpers.load_methods('mmseg/models/uda/dacs.py', 'DACS', ['train_step'])
        model = cls()
        cls.__call__ = lambda self, **kw: {'decode.loss_seg': 1.0,
            'loss_phy_src': 0.7, 'doris.loss_edge_pos': 3.0, 'decode.edge_diag.pos': 0.2}
        model.lambda_phy_src = 0.08
        optimizer = SimpleNamespace(zero_grad=lambda: None, step=lambda: None)
        result = model.train_step(dict(img_metas=[{}]), optimizer)
        self.assertAlmostEqual(result['log_vars']['loss'], 1.056)


@unittest.skipIf(torch is None, 'Torch unavailable')
class PaperMechanismTests(unittest.TestCase):
    def test_tau_alias_conflicts_and_legacy_defaults(self):
        cls = helpers.component('mmseg/models/uda/proto_gate.py', 'ProtoGate')
        legacy = cls()
        self.assertEqual((legacy.tau_proto, legacy.tau_conf_proto), (0.70, 0.90))
        self.assertEqual(cls(target_weight_thr=0.4).tau_proto, 0.4)
        self.assertEqual(cls(tau_proto=0.25, target_weight_thr=0.25).tau_proto, 0.25)
        with self.assertRaises(ValueError):
            cls(tau_proto=0.25, target_weight_thr=0.4)
        with self.assertRaises(ValueError):
            cls(tau_conf_proto=0.88, target_conf_thr=0.9)

    def test_tau_selection_counts_min_pixels_and_warmup(self):
        cls = helpers.component('mmseg/models/uda/proto_gate.py', 'ProtoGate')
        gate = cls(tau_proto=0.25, tau_conf_proto=0.88, min_pixels=8,
                   target_update_start=8, separate_domain_prototypes=True)
        gate.collect_diagnostics = True
        feat = torch.ones(1, 4, 2, 8)
        label = torch.zeros(1, 2, 8).long()
        conf = torch.ones_like(label).float()
        weight = torch.zeros_like(conf)
        weight.reshape(-1)[:8] = 0.25
        conf.reshape(-1)[7] = 0.87
        gate.update_target(feat, label, conf, weight, iter=8)
        self.assertFalse(gate.target_valid_proto[0])
        self.assertEqual(gate.last_target_diagnostics['class_0.selected_count'], 7)
        self.assertEqual(gate.last_target_diagnostics['class_0.updated_count'], 0)
        conf.reshape(-1)[7] = 0.88
        gate.update_target(feat, label, conf, weight, iter=7)
        self.assertEqual(gate.last_target_diagnostics['class_0.selected_count'], 8)
        self.assertEqual(gate.last_target_diagnostics['class_0.updated_count'], 0)
        gate.update_target(feat, label, conf, weight, iter=8)
        self.assertTrue(gate.target_valid_proto[0])
        self.assertEqual(gate.last_target_diagnostics['class_0.updated_count'], 8)

    def test_new_pair_response_has_no_legacy_power_or_ramp(self):
        from test_doris_stage1 import new_model
        model = new_model()
        model.local_iter = 0
        model.struct_edge_pair_power = 0.5
        model.struct_edge_pair_prior_ready.fill_(True)
        model.struct_edge_pair_prior.fill_(0.25)
        labels = torch.zeros(1, 3, 4).long()
        labels[:, :, 2:] = 1
        response = model._build_boundary_pair_response(labels)
        torch.testing.assert_close(response[:, :, :, 1:3],
                                   torch.full((1, 1, 3, 2), 0.25))
        self.assertEqual(response[:, :, :, 0].sum(), 0)

    def test_edge_components_sum_to_single_loss_and_keep_gradient(self):
        from test_doris_stage1 import edge_loss
        cls = edge_loss()['StructEdgeLoss']
        loss = cls(edge_radius=1, boundary_loss_mode='doris_balanced', loss_weight=0.1)
        loss.collect_diagnostics = True
        logits = torch.randn(1, 5, 12, 12, requires_grad=True)
        label = torch.zeros(1, 12, 12).long()
        label[:, :, 6:] = 1
        value = loss(logits, label, struct_edge_weight=torch.full((1, 1, 12, 12), 0.003))
        diag = loss.last_diagnostics
        torch.testing.assert_close(value.detach(), diag['pos'] + diag['neg'] + diag['ce'])
        self.assertGreater(diag['positive_count'], 0)
        self.assertGreater(diag['negative_count'], 0)
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum(), 0)


if __name__ == '__main__':
    unittest.main()
