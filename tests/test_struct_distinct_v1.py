"""Focused Structural V2 production and historical compatibility checks."""
import importlib.util
import math
from pathlib import Path
import runpy
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.struct_schedule import struct_gate_strength

try:
    import torch
except ImportError:
    torch = None


class ScheduleTests(unittest.TestCase):
    def test_early4k_and_legacy(self):
        for step, expected in zip((0, 1000, 2000, 3000, 4000, 6000, 12000), (.1, .325, .55, .775, 1., 1., 1.)):
            self.assertAlmostEqual(struct_gate_strength(step, 'early_nonzero_linear'), expected)
        for step, expected in ((0, 0), (4000, 0), (6000, 0), (8000, .5), (10000, 1)):
            self.assertEqual(struct_gate_strength(step), expected)
        for kwargs in (dict(schedule='bad'), dict(schedule='early_nonzero_linear', full_iter=0),
                       dict(schedule='early_nonzero_linear', full_iter=math.nan),
                       dict(schedule='early_nonzero_linear', initial_strength=math.nan)):
            with self.assertRaises(ValueError):
                struct_gate_strength(0, **kwargs)


@unittest.skipIf(torch is None, 'Torch CPU is required for production Structural components')
class DistinctTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Gate = runpy.run_path(str(ROOT/'mmseg/models/uda/struct_prior_gate.py'))['StructPriorConsistencyGate']
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def gate(self, **kwargs):
        cfg = dict(fusion_mode='struct_distinct_v1', adjacency_mode='cross_class_conditional_v1',
            semantic_score_mode='probability_margin_v1', confidence_weight=.15, continuity_weight=.30,
            adjacency_weight=.30, semantic_weight=.25, struct_gate_schedule='early_nonzero_linear',
            struct_min_weight=.65, struct_filter_threshold=.32)
        cfg.update(kwargs)
        return self.Gate(5, **cfg)

    def test_structural_mparc_full_use_identical_production_gate(self):
        from test_minimal_output import static_config
        prefix = 'configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi'
        gates = [self.Gate(5, **static_config(prefix+suffix+'.py')['uda']['struct_cfg'])
                 for suffix in ('_struct_gate', '_phy_struct_gate', '_proto_struct_edge')]
        probabilities = torch.full((1, 5, 3, 3), .2)
        label = torch.tensor([[[0, 1, 2], [3, 4, 2], [0, 1, 2]]])
        for step, expected in zip((0, 1000, 2000, 3000, 4000, 6000, 12000), (.1, .325, .55, .775, 1., 1., 1.)):
            outputs = []
            for gate in gates:
                self.assertAlmostEqual(gate.gate_strength(step), expected)
                self.assertEqual(gate.gate_strength(step), gate.feature_strength(step, 6000, 4000))
                weight, response = gate.apply(target_probs=probabilities, pseudo_label=label,
                    pseudo_weight=torch.ones_like(label).float(), iter=step, update_priors=False)
                outputs.append((weight, response['R_struct']))
            for output in outputs[1:]:
                torch.testing.assert_close(output[0], outputs[0][0], rtol=0, atol=0)
                torch.testing.assert_close(output[1], outputs[0][1], rtol=0, atol=0)

    def test_continuity_edges_ignore_and_old_modes(self):
        gate = self.Gate(5)
        label = torch.full((1, 3, 3), 2)
        self.assertTrue(torch.all(gate.compute_continuity_score(label) == 1))
        label[0, 0, 0] = 255
        score = gate.compute_continuity_score(label)
        self.assertEqual(score[0, 0, 0], 0)
        self.assertEqual(score[0, 1, 1], 1)
        label[0, 1, 2] = 4
        self.assertAlmostEqual(score[0, 1, 1].item(), 1.)
        self.assertAlmostEqual(gate.compute_continuity_score(label)[0, 1, 1].item(), 6/7)
        gate.source_adj_initialized[:] = True
        gate.source_adj_ema.copy_(torch.eye(5))
        local = gate.compute_continuity_score(label)
        torch.testing.assert_close(gate.compute_adjacency_score(label)[label != 255], local[label != 255])
        for mode in ('legacy_product', 'doris_confidence_guided'):
            historical = self.Gate(5, fusion_mode=mode)
            historical.load_state_dict(gate.state_dict(), strict=True)
            probabilities = torch.rand(1, 5, 3, 3).softmax(1)
            _, response = historical.apply(target_probs=probabilities, pseudo_label=label,
                pseudo_weight=torch.ones(1, 3, 3), iter=8000, collect_debug=True)
            confidence = probabilities.max(1)[0]
            if mode == 'legacy_product':
                expected = confidence*response['R_continuity']*response['R_adj']*response['R_sem'].sqrt()
            else:
                expected = historical.fuse_doris_responses(confidence, response['R_continuity'],
                    response['R_adj'], response['R_sem'], label)
            torch.testing.assert_close(response['R_struct'], expected)

    def test_cross_adjacency_availability_and_mismatch(self):
        gate = self.gate(adj_cross_uniform_smoothing=0.)
        gate.source_adj_initialized[2] = True
        gate.source_adj_ema[2, 2] = .9
        gate.source_adj_ema[2, 4] = .1
        label = torch.full((1, 3, 3), 4)
        label[0, 1, 1] = 2
        score, details = gate.compute_adjacency_score(label, return_details=True)
        self.assertAlmostEqual(score[0, 1, 1].item(), 1.)
        self.assertTrue(details['adjacency_available'][0, 1, 1])
        self.assertEqual(details['target_cross_neighbor_count'][0, 1, 1], 8)
        gate.source_adj_ema[2, 4] = 0
        gate.source_adj_ema[2, 1] = .1
        self.assertEqual(gate.compute_adjacency_score(label)[0, 1, 1], 0)
        for setup in ('interior', 'uninitialized', 'insufficient_mass'):
            if setup == 'interior': label.fill_(2)
            if setup == 'uninitialized': gate.source_adj_initialized[:] = False
            if setup == 'insufficient_mass':
                label.fill_(4)
                label[0, 1, 1] = 2
                gate.source_adj_initialized[2] = True
                gate.source_adj_ema[2, 1] = .001
            score, details = gate.compute_adjacency_score(label, return_details=True)
            self.assertFalse(details['adjacency_available'][0, 1, 1])
            self.assertEqual(score[0, 1, 1], 1)
            self.assertTrue(torch.isfinite(score).all())

    def test_probability_margin_temperature_and_missing_prototypes(self):
        gate = self.gate()
        gate.semantic_proto_ema.copy_(torch.eye(5))
        gate.semantic_proto_initialized[:] = True
        label = torch.full((1, 1, 1), 2)
        probabilities = torch.tensor([.1, .1, .5, .1, .2]).reshape(1, 5, 1, 1)
        score, detail = gate.compute_semantic_score(probabilities, label, return_details=True)
        self.assertGreater(score.item(), .5)
        self.assertTrue(detail['semantic_available'].item())
        gate.semantic_temperature = .05
        self.assertGreater(gate.compute_semantic_score(probabilities, label).item(), score.item())
        probabilities[0, 2] = .2
        self.assertAlmostEqual(gate.compute_semantic_score(probabilities, label).item(), .5)
        probabilities[0, 4] = .6
        self.assertLess(gate.compute_semantic_score(probabilities, label).item(), .5)
        gate.semantic_proto_initialized[:] = False
        gate.semantic_proto_initialized[2] = True
        _, detail = gate.compute_semantic_score(probabilities, label, return_details=True)
        self.assertFalse(detail['semantic_available'].item())

    def test_four_way_fusion_and_invalid_parameters(self):
        gate = self.gate()
        values = [torch.full((1, 2, 2), v) for v in (.2, .8, .6, .4)]
        available = [torch.ones_like(values[0], dtype=torch.bool) for _ in range(4)]
        score, _ = gate.fuse_distinct_responses(values, available)
        self.assertAlmostEqual(score[0, 0, 0].item(), (.15*.2+.3*.8+.3*.6+.25*.4)/(1+gate.eps), places=6)
        available[2].fill_(False)
        score, _ = gate.fuse_distinct_responses(values, available)
        self.assertAlmostEqual(score[0, 0, 0].item(), (.15*.2+.3*.8+.25*.4)/(.7+gate.eps), places=6)
        for kwargs in (dict(confidence_weight=-.1), dict(confidence_weight=.5),
                       dict(confidence_weight=math.nan), dict(semantic_weight=math.inf),
                       dict(semantic_temperature=0), dict(adj_cross_uniform_smoothing=2)):
            with self.assertRaises(ValueError): self.gate(**kwargs)

    def test_gate_and_feature_early_schedule_and_readonly_priors(self):
        gate = self.gate()
        before = {k: v.clone() for k, v in gate.state_dict().items()}
        probabilities = torch.full((1, 5, 3, 3), .2)
        label = torch.tensor([[[0, 1, 2], [3, 4, 2], [0, 1, 2]]])
        for step, rho in zip((0, 1000, 2000, 3000, 4000, 6000, 12000), (.1, .325, .55, .775, 1., 1., 1.)):
            self.assertAlmostEqual(gate.gate_strength(step), rho)
            self.assertEqual(gate.feature_strength(step), gate.gate_strength(step))
            weight, response = gate.apply(target_probs=probabilities, pseudo_label=label,
                pseudo_weight=torch.ones_like(label).float(), iter=step, update_priors=False)
            raw = response['R_struct'].clamp_min(.65)
            raw = torch.where(response['R_struct'] < .32, raw*.5, raw)
            expected = torch.where(label >= 2, 1-rho+rho*raw, torch.ones_like(raw))
            torch.testing.assert_close(weight, expected)
        for key in before: torch.testing.assert_close(gate.state_dict()[key], before[key])
        gate.struct_gate_schedule = 'legacy_warmup_linear'
        self.assertEqual(gate.gate_strength(4000), 0)
        self.assertEqual(gate.feature_strength(8000), .5)

    def test_differentiable_response_has_finite_gradients(self):
        gate = self.gate(detach_gate=False)
        gate.semantic_proto_ema.copy_(torch.eye(5))
        gate.semantic_proto_initialized[:] = True
        logits = torch.randn(1, 5, 3, 3, requires_grad=True)
        probabilities = logits.softmax(1)
        label = torch.full((1, 3, 3), 2)
        weight, _ = gate.apply(target_probs=probabilities, pseudo_label=label,
            pseudo_weight=torch.ones_like(label).float(), iter=4000, update_priors=False)
        weight.sum().backward()
        self.assertTrue(torch.isfinite(weight).all())
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
