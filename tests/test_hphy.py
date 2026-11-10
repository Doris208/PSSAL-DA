"""CPU tensor/DACS component regression tests, distinct from full-model GPU smoke."""
import copy
import importlib.util
import json
import subprocess
import unittest

import numpy as np

import test_minimal_output as helpers

torch = helpers.torch
ROOT = helpers.ROOT


@unittest.skipIf(torch is None, 'PyTorch is required')
class HPhyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dacs = helpers.dacs_component()
        cls.scope = cls.dacs.__init__.__globals__
        torch.set_num_threads(2)

    def model(self, suffix='_phy_gate'):
        cfg = helpers.static_config('configs/hrda/doris/goji26RGB2goji30RGB/'
            'doris_goji26RGB2goji30RGB_hrda_3vi' + suffix + '.py')
        return self.dacs(**cfg['uda'], model=cfg['model'], max_iters=15000)

    def test_head_probability_decomposition_and_source_gradients(self):
        model = self.model()
        response = model.phy_head(torch.rand(2, 3, 32, 32))
        for value in response.values():
            self.assertEqual(value.shape, (2, 1, 32, 32))
            self.assertTrue(torch.isfinite(value).all())
        total = response['p_goji'] + response['p_other_veg'] + response['p_nonveg']
        torch.testing.assert_close(total, torch.ones_like(total))
        for key in ('r_veg', 'q_goji_given_veg', 'p_goji', 'p_other_veg', 'p_nonveg'):
            self.assertTrue(response[key].ge(0).all() and response[key].le(1).all())
        gt = (torch.arange(2048).reshape(2, 1, 32, 32) % 6).long()
        gt[gt == 5] = 255
        logs = {}
        _, loss = model._compute_phy_src_loss(torch.rand(2, 3, 32, 32), gt, logs)
        loss.backward()
        self.assertGreater(model.phy_head.gv_head.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.model.weight.grad)
        self.assertAlmostEqual(logs['loss_phy_src_weighted'], .08 * logs['loss_phy_src_raw'], places=7)

    def test_source_mapping_balancing_missing_groups_and_ignore(self):
        for labels in ([0, 1, 2, 3, 4, 255], [255] * 6, [2] * 6, [0] * 6, [1] * 6, [0, 1, 1, 1, 1, 1]):
            gt = torch.tensor(labels).reshape(1, 1, 1, 6)
            veg = torch.zeros(1, 1, 1, 6, requires_grad=True)
            gv = torch.zeros_like(veg, requires_grad=True)
            loss, parts = self.scope['hierarchical_source_loss'](
                dict(logit_veg=veg, logit_gv=gv), gt)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            excluded = (gt > 1) | (gt < 0)
            self.assertTrue(gv.grad[excluded].eq(0).all())
            self.assertTrue(veg.grad[gt == 255].eq(0).all())
            if gt.eq(0).any():
                self.assertTrue(gv.grad[gt == 0].lt(0).all())
            if gt.eq(1).any():
                self.assertTrue(gv.grad[gt == 1].gt(0).all())
            if gt.eq(0).any() and gt.eq(1).any():
                self.assertAlmostEqual(gv.grad[gt == 0].sum().abs().item(),
                                       gv.grad[gt == 1].sum().abs().item(), places=6)

    def test_conditional_selection_ramp_continuity_and_zero_weights(self):
        label = torch.tensor([[[0, 1, 2, 255, 0]]])
        response = dict(p_goji=torch.full((1, 1, 1, 5), .72, requires_grad=True),
                        p_other_veg=torch.full((1, 1, 1, 5), .08, requires_grad=True))
        before = torch.tensor([[[.6, .6, .6, .6, 0.]]])
        for iteration, rho in ((0, 0), (6000, 0), (7000, .25), (10000, 1), (15000, 1)):
            after, gate, score = self.scope['hierarchical_phy_gate'](response, label, before, iteration)
            self.assertFalse(after.requires_grad)
            self.assertAlmostEqual(score[0, 0, 0].item(), .72, places=6)
            self.assertAlmostEqual(score[0, 0, 1].item(), .08, places=6)
            self.assertAlmostEqual(gate[0, 0, 0].item(), 1 - rho * .25 * .28, places=6)
            self.assertAlmostEqual(gate[0, 0, 1].item(), 1 - rho * .25 * .92, places=6)
            self.assertTrue(gate[..., 2:].eq(1).all())
            self.assertTrue((after <= before).all() and after.ge(0).all())
            self.assertEqual(after[..., -1].item(), 0)
        head = self.model().phy_head
        head.eval()
        with torch.no_grad():
            head.gv_head.weight.zero_()
            head.gv_head.bias.fill_(-2)
            low = head(torch.ones(1, 3, 8, 8))
            head.gv_head.bias.fill_(2)
            high = head(torch.ones(1, 3, 8, 8))
        torch.testing.assert_close(low['r_veg'], high['r_veg'])
        self.assertTrue((high['p_goji'] > low['p_goji']).all())
        self.assertTrue((high['p_other_veg'] < low['p_other_veg']).all())

    def test_early4k_schedule_at_required_iterations_and_fixed_response(self):
        strength = self.scope['phy_gate_strength']
        label = torch.tensor([[[0, 1, 2, 255, 0]]])
        response = dict(p_goji=torch.full((1, 1, 1, 5), .4, requires_grad=True),
                        p_other_veg=torch.full((1, 1, 1, 5), .4, requires_grad=True))
        weight = torch.tensor([[[.6, .6, .6, .6, 0.]]])
        for iteration, expected in ((0, .1), (500, .2125), (1000, .325), (1500, .4375),
                                    (2000, .55), (2500, .6625), (3000, .775),
                                    (4000, 1.), (7000, 1.), (10000, 1.), (15000, 1.)):
            rho = strength(iteration, schedule='early_nonzero_linear', full_iter=4000)
            self.assertAlmostEqual(rho, expected)
            details = self.scope['hierarchical_phy_gate'](response, label, weight,
                iteration, schedule='early_nonzero_linear', full_iter=4000, return_details=True)
            self.assertEqual(details['rho'], rho)
            torch.testing.assert_close(details['A_phy'][..., :2], torch.full((1, 1, 2), .15*rho))
            self.assertTrue(details['G_phy'][..., 2:].eq(1).all())
            torch.testing.assert_close(details['W_after'], weight*details['G_phy'])
            positive = weight > 0
            torch.testing.assert_close(details['A_phy'][positive],
                                       1-details['W_after'][positive]/weight[positive])
            self.assertFalse(details['W_after'].requires_grad)
        for initial, full in ((0., 4000), (1.1, 4000), (.1, 0), (.1, -1), (.1, float('inf'))):
            with self.assertRaises(ValueError):
                strength(0, schedule='early_nonzero_linear', initial_strength=initial, full_iter=full)
        for floor in (0, -1, 1.1):
            with self.assertRaises(ValueError):
                self.scope['hierarchical_phy_gate'](response, label, weight, 0, min_weight=floor)
        with self.assertRaises(ValueError):
            strength(0, schedule='unknown')

    def test_dacs_uses_actual_local_iter_and_snapshots_survive_classmix_mutation(self):
        model = self.model()
        model.eval()
        model.local_iter = 2000
        before = torch.full((1, 8, 8), .6)
        label = torch.zeros_like(before).long()
        logs = {}
        response, after, gate = model._apply_phy_gate(torch.rand(1, 3, 8, 8), label, before, logs)
        self.assertEqual(logs['phy_gate_iter'], 2000)
        self.assertAlmostEqual(logs['phy_gate_rho'], .55)
        torch.testing.assert_close(1-gate, .25*.55*(1-response['s_phy_class_agreement'][:, 0]))
        snapshot = response['w_after_phy'].clone()
        after.fill_(1)  # The real ClassMix loop writes into the returned weight tensor.
        torch.testing.assert_close(response['w_after_phy'], snapshot)
        torch.testing.assert_close(response['w_before_phy'][:, 0], before)
        self.assertEqual(logs['phy_gate_active_ratio'], 1)

    def test_saved_schedule_restores_legacy_and_early3k_without_changing_heads(self):
        requested = self.model().phy_cfg
        restore = self.scope['restore_phy_gate_config']
        parameters = self.scope['phy_gate_parameters']
        old = dict(requested, phy_gate_full_iter=3000, phy_rampup_iters=3000)
        restored = restore(requested, old)
        self.assertEqual(parameters(restored)['full_iter'], 3000)
        historical = dict(mode='hierarchical_gv', phy_warmup_iters=6000, phy_rampup_iters=4000)
        restored = restore(requested, historical)
        self.assertEqual(parameters(restored), dict(schedule='legacy_warmup_linear', warmup=6000, rampup=4000))
        self.assertEqual(restored['phy_min_weight'], .75)
        self.assertEqual(requested['phy_gate_full_iter'], 4000)

    def test_target_gradient_only_vegetation_head_and_shared_encoder(self):
        model = self.model()
        model.local_iter = 10000
        label = torch.arange(64).reshape(1, 8, 8) % 5
        before = torch.ones(1, 8, 8)
        before[:, 0] = 0
        response, after, gate = model._apply_phy_gate(torch.rand(1, 3, 8, 8), label, before, {})
        segmentation = torch.ones_like(after, requires_grad=True)
        (segmentation * after).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.phy_head.parameters()))
        valid = before.unsqueeze(1)
        loss, _ = self.scope['target_phy_loss'](response, label, torch.ones_like(before), valid,
                                               mode='balanced_high_conf')
        loss.backward()
        self.assertGreater(model.phy_head.veg_head.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.phy_head.gv_head.weight.grad)
        self.assertTrue(gate[:, 0].eq(1).all())

    def test_hphy_diagnostics_survive_minimal_logger_filter(self):
        cls = helpers.load_methods('mmseg/utils/precision_logger.py', 'PrecisionLoggerHook',
                                   ['_keep_train_key'])
        logger = cls()
        logger.output_mode = 'minimal'
        logger.paper_process_logging = False
        for key in ('hphy_goji_pseudo_ratio', 'hphy_goji_agreement_mean',
                    'hphy_vegetation_gate_mean', 'hphy_attenuation_mean_valid_weight',
                    'phy_gate_iter', 'phy_gate_rho', 'phy_gate_active_ratio',
                    'phy_S_mean_active', 'phy_A_mean_active', 'phy_A_max_active',
                    'loss_phy_src', 'loss_phy_tgt'):
            self.assertTrue(logger._keep_train_key(key, None))
        model = self.model()
        label = torch.zeros(1, 8, 8).long()
        logs = {}
        model._apply_phy_gate(torch.rand(1, 3, 8, 8), label,
                              torch.zeros(1, 8, 8), logs, valid_mask=torch.ones(1, 1, 8, 8))
        self.assertEqual(logs['hphy_goji_pseudo_ratio'], 1)
        self.assertEqual(logs['hphy_attenuation_mean_valid_weight'], 0)

    def test_affected_three_configs_losses_gate_backward_and_checkpoint_roundtrip(self):
        for suffix in ('_phy_gate', '_phy_struct_gate', '_proto_struct_edge'):
            model = self.model(suffix)
            model.local_iter = 10000
            gt = torch.arange(128).reshape(2, 1, 8, 8) % 5
            _, source = model._compute_phy_src_loss(torch.rand(2, 3, 8, 8), gt, {})
            response, after, _ = model._apply_phy_gate(torch.rand(2, 3, 8, 8), gt[:, 0], torch.ones(2, 8, 8), {})
            target, _ = self.scope['target_phy_loss'](response, gt[:, 0], torch.ones_like(after),
                                                     mode='balanced_high_conf')
            (source + .008 * target).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.phy_head.parameters()))
            restored = self.model(suffix)
            restored.load_state_dict(model.state_dict())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)
            wrong = copy.deepcopy(model.state_dict())
            wrong.pop('phy_head.gv_head.weight')
            with self.assertRaisesRegex(RuntimeError, 'Incomplete physiology checkpoint'):
                restored.load_state_dict(wrong, strict=False)
            legacy = {'phy_head.layers.0.weight': torch.zeros(1)}
            with self.assertRaisesRegex(RuntimeError, 'mode mismatch'):
                restored.load_state_dict(legacy, strict=False)
            legacy_model = self.model(suffix)
            legacy_model.phy_mode = 'legacy_binary'
            with self.assertRaisesRegex(RuntimeError, 'mode mismatch'):
                legacy_model.load_state_dict(model.state_dict(), strict=False)
            incomplete_hphy = {key: value for key, value in model.state_dict().items()
                               if not key.startswith('phy_head.gv_head.')}
            with self.assertRaisesRegex(RuntimeError, 'mode mismatch'):
                legacy_model.load_state_dict(incomplete_hphy, strict=False)

    def test_online_probe_exports_conditional_maps_and_four_panel_titles(self):
        from test_doris_mechanism import MechanismTests
        from tools.build_doris_mechanism_figures import online_plate
        from matplotlib import pyplot as plt
        model = MechanismTests().model()
        model.phy_mode = 'hierarchical_gv'
        model.phy_min_weight = .75
        model.phy_head = self.scope['HierarchicalPhyReliabilityHead']()
        model.phy_cfg.update(mode='hierarchical_gv', phy_min_weight=.75, phy_filter_threshold=None)
        # Exercise AMP-shaped head responses with fp32 DACS pseudo weights on CPU.
        model.phy_head.register_forward_hook(
            lambda module, inputs, output: {key: value.half() for key, value in output.items()})
        model.enable_struct = model.enable_proto = model.enable_struct_edge = False
        with torch.no_grad():
            model.ema_model.classifier.weight.zero_()
            model.ema_model.classifier.bias.fill_(-5)
            model.ema_model.classifier.bias[0] = 5
        spec = importlib.util.spec_from_file_location('hphy_observation', ROOT / 'mmseg/utils/mechanism.py')
        observation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observation)
        metadata = dict(iter=10000, sample_id='synthetic', synthetic_fixture=True,
                        modules=dict(phy=True, struct=False, proto=False, edge=False),
                        states=dict(phy='active'), capture_mode='synthetic_test')
        payload = observation.branch_probe_payload(model, torch.rand(1, 6, 16, 16),
            torch.rand(1, 3, 16, 16), [dict(img_norm_cfg=dict(mean=[0]*6, std=[1]*6))],
            torch.ones(1, 1, 16, 16), metadata)
        for key in ('r_veg', 'q_goji_given_veg', 'p_goji', 'p_other_veg', 'p_nonveg',
                    's_phy_class_agreement', 'g_phy', 'attenuation_phy',
                    'vi_ndvi', 'vi_ndre', 'vi_cire', 'teacher_label', 'teacher_confidence'):
            self.assertEqual(payload[key].shape, (16, 16))
        np.testing.assert_allclose(payload['w_after_phy'], payload['w_before_phy'] * payload['g_phy'])
        np.testing.assert_allclose(payload['attenuation_phy'], 1 - payload['g_phy'])
        self.assertEqual(json.loads(str(payload['metadata']))['head_version'], 'hierarchical_gv')
        figure = online_plate([payload], 'phys')
        try:
            titles = [axis.get_title() for axis in figure.axes]
            self.assertIn('(c) Class-conditional agreement S_phy', titles)
            self.assertIn('(d) Relative attenuation A_phy = 1-G_phy', titles)
        finally:
            plt.close(figure)

    def test_disabled_configs_and_non_phy_parameters_are_unchanged(self):
        paths = sorted((ROOT / 'configs').rglob('doris_*.py'))
        disabled = 0
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            cfg = helpers.static_config(relative)
            if not cfg.get('uda', {}).get('phy_cfg', {}).get('enable', False):
                old = subprocess.check_output(['git', 'show', 'ef089d4:' + relative], cwd=ROOT)
                self.assertEqual(path.read_text(encoding='utf-8'),
                                 old.decode('utf-8').replace('\r\n', '\n'))
                disabled += 1
        self.assertEqual(disabled, 20)
        for _, suffix, _ in __import__('tools.check_doris_hphy_configs', fromlist=['EXPERIMENTS']).EXPERIMENTS:
            model = self.model(suffix)
            self.assertEqual(model.phy_head is not None, suffix in ('_phy_gate', '_phy_struct_gate', '_proto_struct_edge'))


class HPhyEvidenceTests(unittest.TestCase):
    def test_auc_ties_perfect_reverse_and_missing_groups(self):
        from tools.doris_phy_evidence import error_auc
        errors = np.array([False, True, False, True])
        self.assertEqual(error_auc(errors, np.ones(4))['roc_auc'], .5)
        self.assertEqual(error_auc(errors, errors.astype(float))['roc_auc'], 1)
        self.assertEqual(error_auc(errors, 1 - errors.astype(float))['roc_auc'], 0)
        self.assertIsNone(error_auc(np.zeros(4, dtype=bool), np.ones(4))['pr_auc'])

    def test_offline_four_groups_multi_tile_auc_and_budget(self):
        from tools.doris_phy_evidence import hphy_reliability_statistics, matched_budget_retention
        retained = matched_budget_retention(np.array([False, True]), np.array([0., 1.]),
                                            .4, np.array([.2, .8]))
        self.assertEqual(retained['correct_retained'], 1)
        self.assertAlmostEqual(retained['error_reduction'], .75)
        score = np.array([[.9, .1], [.1, .9]])
        gate = 1 - .25 * (1 - score)
        payload = dict(reference_label=np.array([[0, 0], [1, 1]]),
            pseudo_label=np.array([[0, 1], [0, 1]]), valid_mask=np.ones((2, 2), bool),
            w_before_phy=np.full((2, 2), .6), w_conf=np.full((2, 2), .98),
            r_veg=np.ones((2, 2)), q_goji_given_veg=np.array([[.9, .9], [.1, .1]]),
            s_phy_class_agreement=score, g_phy=gate, attenuation_phy=1-gate)
        report = hphy_reliability_statistics([payload, payload], bootstrap_samples=10)
        self.assertTrue(all(group['valid_pixels'] == 2 for group in report['groups']))
        for row in report['error_detection']:
            self.assertEqual(row['hphy']['roc_auc'], 1)
            self.assertEqual(row['teacher_uncertainty']['roc_auc'], .5)
            self.assertEqual(row['hphy']['bootstrap_valid_replicates'], 10)
            self.assertAlmostEqual(row['actual_hphy_weight_retention']['correct_retained'], .975)


if __name__ == '__main__':
    unittest.main()
