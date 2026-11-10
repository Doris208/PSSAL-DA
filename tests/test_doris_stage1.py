"""DORIS fusion/loss safety checks on CPU; real MMCV config check is optional."""
import ast
import base64
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
import random
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_minimal_output as helpers
from test_minimal_output import ROOT, torch

HAS_BASELINE = (ROOT / 'outputs/doris_stage1_before.json').is_file()


def new_model(direction='goji26IRRG2goji30RGB'):
    cfg = helpers.static_config('configs/hrda/doris/' + direction + '/doris_' + direction +
                                '_hrda_3vi_proto_struct_edge.py')
    cls = helpers.dacs_component()
    model = cls(**cfg['uda'], model=cfg['model'], max_iters=20000)
    model.local_iter = 16000
    return model


def stage_source(name):
    snapshot = ROOT / 'outputs/doris_stage1_before.json'
    if snapshot.is_file():
        files = json.loads(snapshot.read_text(encoding='utf8'))['files']
        if name in files:
            return base64.b64decode(files[name]).decode('utf-8-sig')
    return (ROOT / name).read_text(encoding='utf-8-sig')


def edge_loss(original=False):
    path = 'mmseg/models/losses/struct_edge_loss.py'
    tree = ast.parse(stage_source(path) if original else (ROOT / path).read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional,
                 LOSSES=SimpleNamespace(register_module=lambda: lambda cls: cls))
    exec(compile(tree, path, 'exec'), scope)
    return scope


@unittest.skipIf(torch is None, 'Torch CPU environment unavailable')
class DorisCoreTests(unittest.TestCase):
    def test_full_dacs_train_step_with_cpu_standins_keeps_masks_and_legacy_updates(self):
        import numpy as np
        module = edge_loss()
        class Student(torch.nn.Module):
            def __init__(self, boundary_loss):
                super().__init__()
                self.features = torch.nn.Conv2d(6, 4, 1)
                self.classifier = torch.nn.Conv2d(4, 5, 1)
                self.boundary_loss = boundary_loss
                self.decode_head = SimpleNamespace(debug=False, debug_output={})
                self.received_valid = []
            def extract_feat(self, image):
                return [self.features(image)]
            def generate_pseudo_label(self, image, metas):
                return self.classifier(self.features(image))
            def forward_train(self, image, metas, label, weight=None,
                              struct_edge_weight=None, struct_prior_response=None,
                              return_feat=False, struct_edge_valid_mask=None):
                feature = self.features(image)
                logits = self.classifier(feature)
                semantic_label = label[:, 0] if label.dim() == 4 else label
                semantic = torch.nn.functional.cross_entropy(logits, semantic_label, reduction='none')
                if weight is not None:
                    semantic = semantic * weight
                losses = {'decode.loss_seg': semantic.mean(),
                          'decode.loss_struct_edge': self.boundary_loss(logits, label,
                              struct_edge_weight=struct_edge_weight,
                              struct_edge_valid_mask=struct_edge_valid_mask)}
                if return_feat:
                    losses['features'] = [feature]
                if struct_edge_valid_mask is not None:
                    self.received_valid.append(struct_edge_valid_mask.detach().clone())
                return losses
        def strong_transform(params, data=None, target=None):
            mask = params['mix'][0]
            mixed_data = ((mask * data[0] + (1 - mask) * data[1]).unsqueeze(0)
                          if data is not None else None)
            mixed_target = ((mask * target[0] + (1 - mask) * target[1]).unsqueeze(0)
                            if target is not None else None)
            return mixed_data, mixed_target
        original_legacy = helpers.static_config(
            'configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py')
        results = []
        observed_results = []
        for original, doris, observe in ((True, False, False), (False, False, False),
                                        (False, True, False), (False, True, True)):
            if original and not HAS_BASELINE:
                continue
            cfg = (helpers.static_config('configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py')
                   if doris else copy.deepcopy(original_legacy))
            cfg['uda']['mechanism_logging'] = observe
            cfg['uda']['mechanism_sample_interval'] = 1
            with patch.object(helpers, 'original_source', stage_source):
                cls = helpers.dacs_component(original=original)
            torch.manual_seed(81)
            model = cls(**cfg['uda'], model=cfg['model'], max_iters=20000)
            loss_cfg = dict(cfg['model']['decode_head']['loss_decode'][1])
            loss_cfg.pop('type')
            loss = module['StructEdgeLoss'](**loss_cfg)
            model.model = Student(loss)
            model.ema_model = copy.deepcopy(model.model)
            model._init_ema_weights()
            model.local_iter = 16000
            cls.forward = lambda self, **kw: self.forward_train(**kw)
            parser = helpers.load_methods('mmseg/models/segmentors/base.py',
                'BaseSegmentor', ['_parse_losses'],
                dict(OrderedDict=OrderedDict, torch=torch, dist=torch.distributed))
            cls._parse_losses = staticmethod(parser._parse_losses)
            scope = model.forward_train.__globals__
            if observe:
                import importlib.util
                spec = importlib.util.spec_from_file_location('mechanism_fixture',
                    ROOT / 'mmseg/utils/mechanism.py')
                observation = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(observation)
                scope['training_record'] = observation.training_record
                scope['time'] = time
            mask = torch.zeros(1, 32, 32).long()
            mask[:, :, :16] = 1
            scope.update(np=np, random=random,
                _DropoutNd=torch.nn.modules.dropout._DropoutNd, DropPath=torch.nn.Identity,
                get_mean_std=lambda metas, dev: (torch.zeros(len(metas), 6, 1, 1),
                                                 torch.ones(len(metas), 6, 1, 1)),
                convert_to_norm=lambda image, *args: image,
                get_class_masks=lambda gt: [mask.clone() for _ in range(len(gt))],
                strong_transform=strong_transform,
                add_prefix=lambda values, prefix: {prefix + '.' + k: v for k, v in values.items()})
            phy_tree = ast.parse((ROOT / 'mmseg/models/uda/phy_reliability.py').read_text())
            fns = [n for n in phy_tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('masked_bce_loss', 'target_phy_loss')]
            exec(compile(ast.Module(body=fns, type_ignores=[]), 'phy', 'exec'), scope)
            data = dict(img=torch.randn(1, 6, 32, 32), target_img=torch.randn(1, 6, 32, 32),
                        img_metas=[{}], target_img_metas=[{}],
                        source_vi_img=torch.randn(1, 3, 32, 32),
                        target_vi_img=torch.randn(1, 3, 32, 32),
                        gt_semantic_seg=torch.zeros(1, 1, 32, 32).long(),
                        valid_pseudo_mask=torch.ones(1, 1, 32, 32))
            data['gt_semantic_seg'][:, :, :, 16:] = 1
            data['valid_pseudo_mask'][:, :, :3] = 0
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            output = model.train_step(data, optimizer)
            self.assertTrue(all(torch.isfinite(torch.tensor(value))
                                for value in output['log_vars'].values()))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            self.assertEqual(model.local_iter, 16001)
            if doris:
                observed_results.append(({key: value.detach().clone()
                    for key, value in model.state_dict().items()},
                    {name: value.grad.clone() for name, value in model.named_parameters()
                     if value.grad is not None}))
                self.assertEqual(len(model.mechanism_records), int(observe))
                self.assertEqual(len(model.get_model().received_valid), 1)
                support = model.get_model().received_valid[0]
                self.assertFalse(support[:, :, :, 15:17].any())
                self.assertFalse(support[:, :, :3, 25:].any())
            elif HAS_BASELINE:
                results.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        if HAS_BASELINE:
            for key, value in results[0].items():
                torch.testing.assert_close(value, results[1][key], rtol=0, atol=0)
        for before, after in zip(observed_results[0], observed_results[1]):
            for key, value in before.items():
                torch.testing.assert_close(value, after[key], rtol=0, atol=0)

    def test_hrda_crop_and_head_plumbing_keep_valid_mask_aligned(self):
        captured = []
        class Parent:
            def losses(self, logits, label, seg_weight=None, struct_edge_weight=None,
                       struct_edge_valid_mask=None):
                captured.append((label.clone(), struct_edge_valid_mask.clone()))
                return {'loss_seg': torch.tensor(1.0)}
        tree = ast.parse((ROOT / 'mmseg/models/decode_heads/hrda_head.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HRDAHead')
        cls.bases, cls.decorator_list = [ast.Name(id='Parent', ctx=ast.Load())], []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'losses']
        def crop(tensor, box):
            y1, y2, x1, x2 = box
            return tensor[..., y1:y2, x1:x2]
        scope = dict(Parent=Parent, crop=crop,
                     add_prefix=lambda data, prefix: {prefix + '.' + k: v for k, v in data.items()})
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'head', 'exec'), scope)
        head = scope['HRDAHead']()
        head.hr_loss_weight, head.lr_loss_weight, head.enable_hr_crop = 0.1, 0.1, True
        head.hr_crop_box, head.debug_output = (2, 8, 3, 10), {}
        label = torch.arange(144).reshape(1, 1, 12, 12)
        valid = label.remainder(3).eq(0)
        head.losses((None, None, None), label, struct_edge_valid_mask=valid)
        self.assertEqual(len(captured), 3)
        torch.testing.assert_close(captured[0][1], valid)
        torch.testing.assert_close(captured[1][1], valid)
        torch.testing.assert_close(captured[2][1], valid[..., 2:8, 3:10])
        self.assertEqual(captured[2][0].shape, captured[2][1].shape)

    def test_export_restores_priors_uses_teacher_and_never_updates_state(self):
        from tools.export_doris_response_maps import compute_responses, load_response_checkpoint
        model = new_model()
        model.struct_gate.source_adj_initialized.fill_(True)
        model.struct_gate.semantic_proto_initialized.fill_(True)
        model.struct_gate.source_adj_ema.copy_(torch.eye(5))
        model.struct_gate.semantic_proto_ema.copy_(torch.eye(5))
        model.struct_edge_pair_prior.fill_(0.5)
        model.struct_edge_pair_prior_ready.fill_(True)
        with self.assertRaisesRegex(ValueError, 'source priors'):
            load_response_checkpoint(new_model(), {'state_dict': {} })
        cold = new_model()
        with self.assertRaisesRegex(ValueError, 'uninitialized'):
            load_response_checkpoint(cold, {'state_dict': cold.state_dict()})
        restored = new_model()
        checkpoint = {'state_dict': model.state_dict()}
        load_response_checkpoint(restored, checkpoint)
        calls = []
        def teacher_predict(image, metas):
            calls.append(1)
            return restored.get_ema_model()(image)
        restored.get_ema_model().generate_pseudo_label = teacher_predict
        before = {k: v.clone() for k, v in restored.state_dict().items()}
        maps = compute_responses(restored, torch.randn(1, 6, 8, 8), [{}])
        self.assertEqual(len(calls), 1)
        for key in ('R_conf', 'R_continuity', 'R_adj', 'R_sem', 'R_struct',
                    'E_prob', 'E_img', 'R_pair', 'R_edge'):
            self.assertEqual(maps[key].shape, (1, 8, 8))
            self.assertTrue(torch.isfinite(maps[key]).all())
        for key, value in before.items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_process_scalars_are_opt_in_and_have_separate_denominators(self):
        model = new_model()
        label = torch.tensor([[[0, 0, 1, 2], [0, 1, 1, 2]]])
        confidence = torch.tensor([[[.99, .5, .99, .8], [.5, .5, .5, .8]]])
        weight = torch.full_like(confidence, .2)
        valid = torch.ones(1, 1, 2, 4).bool()
        valid[:, :, 1] = False
        response = torch.full_like(confidence, .8)
        response[:, :, -1] = .4
        logs = {}
        model._log_doris_process(confidence, label, weight, valid, {'R_struct': response}, logs)
        self.assertEqual(logs, {})
        model.doris_process_logging, model.local_iter = True, 49
        model.doris_log_interval = 50
        model._log_doris_process(confidence, label, weight, valid, {'R_struct': response}, logs)
        self.assertEqual(logs['doris.teacher_spatial_valid_ratio'], .5)
        self.assertEqual(logs['doris.teacher_high_conf_ratio_valid'], .5)
        self.assertEqual(logs['doris.teacher_goji_prediction_ratio_valid'], .5)
        self.assertEqual(logs['doris.teacher_goji_high_conf_ratio_given_class'], .5)
        self.assertAlmostEqual(logs['doris.R_struct_mean_struct_classes_valid'], .4)
        self.assertAlmostEqual(logs['doris.R_struct_mean_all'], .7)
        logger = helpers.load_methods('mmseg/utils/precision_logger.py',
                                      'PrecisionLoggerHook', ['_keep_train_key'])()
        logger.output_mode, logger.paper_process_logging = 'minimal', False
        self.assertTrue(all(logger._keep_train_key(key, None) for key in logs))

    def test_structural_formula_availability_finiteness_and_ignore(self):
        cls = helpers.component('mmseg/models/uda/struct_prior_gate.py',
                                'StructPriorConsistencyGate')
        gate = cls(num_classes=5, fusion_mode='doris_confidence_guided')
        label = torch.tensor([[[0, 1, 2, 255]]])
        conf = torch.full(label.shape, 0.8)
        cont = torch.full(label.shape, 0.2)
        adj = torch.full(label.shape, 0.4)
        sem = torch.full(label.shape, 0.6)
        gate.source_adj_initialized[1:3] = True
        gate.semantic_proto_initialized[2] = True
        result = gate.fuse_doris_responses(conf, cont, adj, sem, label)
        expected = torch.tensor([[[0.8 * 0.2, 0.8 * (0.4 * 0.2 + 0.3 * 0.4) / 0.7,
                                   0.8 * (0.4 * 0.2 + 0.3 * 0.4 + 0.3 * 0.6), 0]]])
        torch.testing.assert_close(result, expected)
        cont[0, 0, 0], adj[0, 0, 1], conf[0, 0, 2] = float('nan'), float('inf'), float('nan')
        result = gate.fuse_doris_responses(conf, cont, adj, sem, label)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((0 <= result) & (result <= 1)).all())
        self.assertEqual(result[0, 0, 0].item(), 0)
        torch.testing.assert_close(result[0, 0, 1], torch.tensor(0.16))

    def test_structure_reuses_confidence_and_ignore_neighbors(self):
        cls = helpers.component('mmseg/models/uda/struct_prior_gate.py',
                                'StructPriorConsistencyGate')
        gate = cls(num_classes=5, fusion_mode='doris_confidence_guided')
        labels = torch.tensor([[[255, 255, 255], [255, 0, 0], [255, 0, 0]]])
        continuity = gate.compute_continuity_score(labels)
        self.assertTrue(continuity[labels.ne(255)].eq(1).all())
        self.assertTrue(continuity[labels.eq(255)].eq(0).all())
        probabilities = torch.zeros(1, 5, 3, 3)
        probabilities[:, 0] = 1
        cached = torch.full_like(continuity, 0.7)
        _, debug = gate.apply(target_probs=probabilities, pseudo_label=labels,
                              pseudo_weight=torch.ones_like(cached), target_confidence=cached,
                              collect_debug=False, log_stats=False)
        torch.testing.assert_close(debug['R_struct'][labels.ne(255)],
                                   torch.full((4,), 0.7))
        self.assertEqual(debug['R_struct'][labels.eq(255)].sum().item(), 0)

    def test_structure_renormalizes_uninitialized_priors(self):
        cls = helpers.component('mmseg/models/uda/struct_prior_gate.py',
                                'StructPriorConsistencyGate')
        gate = cls(num_classes=5, fusion_mode='doris_confidence_guided')
        labels = torch.zeros(1, 3, 3).long()
        probs = torch.zeros(1, 5, 3, 3)
        probs[:, 0] = 0.8
        probs[:, 1] = 0.2
        _, debug = gate.apply(target_probs=probs, pseudo_label=labels,
                              pseudo_weight=torch.ones_like(labels).float(),
                              source_gt=None, collect_debug=False, log_stats=False)
        # Only continuity is evidence; neither default-one prior is ready.
        torch.testing.assert_close(debug['R_struct'], torch.full((1, 3, 3), 0.8))

    def test_pair_is_zero_off_boundaries(self):
        cls = helpers.dacs_component()
        cfg = helpers.static_config(
            'configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py')
        cfg['uda']['struct_edge_cfg']['edge_fusion_mode'] = 'doris_reliable'
        model = cls(**cfg['uda'], model=cfg['model'], max_iters=20000)
        model.local_iter = 12000
        model.struct_edge_pair_prior.fill_(1)
        model.struct_edge_pair_prior_ready.fill_(True)
        response = model._build_boundary_pair_response(torch.zeros(1, 5, 5).long())
        self.assertEqual(response.count_nonzero().item(), 0)

    def test_balanced_loss_preserves_external_scale(self):
        cls = edge_loss()['StructEdgeLoss']
        loss = cls(edge_radius=1, boundary_loss_mode='doris_balanced', ce_weight=0.1)
        torch.manual_seed(5)
        logits = torch.randn(1, 5, 16, 16, requires_grad=True)
        label = torch.zeros(1, 16, 16).long()
        label[:, :, 8:] = 1
        a = loss(logits, label, struct_edge_weight=torch.ones(1, 1, 16, 16))
        b = loss(logits, label, struct_edge_weight=torch.full((1, 1, 16, 16), 0.1))
        torch.testing.assert_close(b, a * 0.1)

    def test_pair_averages_valid_pairs_and_never_wraps_or_uses_ignore(self):
        model = new_model()
        model.struct_edge_pair_power = 1
        model.struct_edge_pair_prior_ready.fill_(True)
        model.struct_edge_pair_prior.zero_()
        model.struct_edge_pair_prior[0, 1] = 0.2
        model.struct_edge_pair_prior[0, 2] = 0.8
        labels = torch.tensor([[[255, 1, 255], [2, 0, 255], [255, 255, 255]]])
        result = model._build_boundary_pair_response(labels)
        torch.testing.assert_close(result[0, 0, 1, 1], torch.tensor(0.5))
        self.assertEqual(result[0, 0][labels[0].eq(255)].sum().item(), 0)
        model.struct_edge_pair_prior_ready.fill_(False)
        self.assertEqual(model._build_boundary_pair_response(labels).sum().item(), 0)
        model.struct_edge_pair_prior_ready.fill_(True)
        labels = torch.zeros(1, 3, 5).long()
        labels[:, :, -1] = 1
        result = model._build_boundary_pair_response(labels)
        self.assertEqual(result[:, :, :, 0].sum().item(), 0)

    def test_edge_formula_source_normalization_and_vi_exclusion(self):
        model = new_model()
        image = torch.randn(1, 6, 9, 9)
        changed_vi = image.clone()
        changed_vi[:, 3:] = torch.randn(1, 3, 9, 9) * 1000
        torch.testing.assert_close(model._build_img_edge_response(image),
                                   model._build_img_edge_response(changed_vi), rtol=0, atol=0)
        model._build_img_edge_response = lambda x: torch.full((1, 1, 9, 9), 0.2)
        model._build_prob_edge_response = lambda x: torch.full((1, 1, 9, 9), 0.8)
        model._build_boundary_pair_response = lambda x: torch.full(
            (1, 1, 9, 9), 0.5 if model.struct_edge_pair_prior_ready.item() else 0.0)
        label = torch.zeros(1, 9, 9).long()
        source = model.doris_edge_responses(image, label)
        torch.testing.assert_close(source['R_edge'], source['E_img'])
        probs = torch.ones(1, 5, 9, 9) / 5
        model.struct_edge_pair_prior_ready.fill_(True)
        fused = model.doris_edge_responses(image, label, probs)['R_edge']
        torch.testing.assert_close(fused, torch.full_like(fused, 0.45 * 0.8 + 0.35 * 0.2 + 0.2 * 0.5))
        model.struct_edge_pair_prior_ready.fill_(False)
        fused = model.doris_edge_responses(image, label, probs)['R_edge']
        torch.testing.assert_close(fused, torch.full_like(fused, 0.45 * 0.8 + 0.35 * 0.2))

    def test_target_reliability_is_smooth_detached_and_independent_of_global_weight(self):
        model = new_model()
        image = torch.randn(1, 6, 8, 8)
        label = torch.zeros(1, 8, 8).long()
        response = torch.ones(1, 1, 8, 8)
        model.doris_edge_responses = lambda *args: {'R_edge': response}
        confidence = torch.full((1, 8, 8), 0.775, requires_grad=True)
        valid = torch.ones(1, 1, 8, 8)
        valid[:, :, :, 0] = 0
        probabilities = torch.ones(1, 5, 8, 8) / 5
        model.local_iter = 2500
        values = []
        for scalar in (0.0, 0.2, 1.0):
            values.append(model._build_struct_edge_weight(
                image, label, 0.002, {}, 'tgt', is_target=True,
                target_probs=probabilities, target_confidence=confidence,
                boundary_valid_mask=valid, base_weight=torch.full_like(confidence, scalar)))
        for value in values:
            self.assertFalse(value.requires_grad)
            torch.testing.assert_close(value, valid * 0.5 * 0.002 * 0.5)
        model.local_iter = 5000
        full = model._build_struct_edge_weight(
            image, label, 0.002, {}, 'tgt', is_target=True,
            target_probs=probabilities, target_confidence=confidence, boundary_valid_mask=valid)
        torch.testing.assert_close(full, values[0] * 2)

    def test_negative_ring_has_actual_false_edge_gradient_and_preserves_support(self):
        module = edge_loss()
        cls = module['StructEdgeLoss']
        loss = cls(edge_radius=1, boundary_loss_mode='doris_balanced', ce_weight=0)
        label = torch.zeros(1, 17, 21).long()
        label[:, :, 10:] = 1
        positive = module['label_to_edge'](label, edge_radius=1)
        # Reliability exists only on positives, so an unpropagated negative term is zero.
        reliability = positive * 0.3
        edge_prediction = torch.full((1, 1, 17, 21), 0.1, requires_grad=True)
        with torch.no_grad():
            edge_prediction[:, :, :, 5:7] = 0.9  # false edge inside the local negative ring
        logits = torch.randn(1, 5, 17, 21, requires_grad=True)
        with patch.dict(loss._doris_balanced_loss.__globals__,
                        prob_to_edge=lambda pred, eps: edge_prediction):
            value = loss(logits, label, struct_edge_weight=reliability)
            value.backward()
        self.assertGreater(edge_prediction.grad[:, :, :, 5:7].sum().item(), 0)
        self.assertLess(edge_prediction.grad[positive.bool()].sum().item(), 0)
        self.assertEqual(edge_prediction.grad[:, :, :, :3].abs().sum().item(), 0)
        # A separate support mask must stop the propagated weights at invalid pixels.
        edge_prediction.grad = None
        valid = torch.ones_like(positive)
        valid[:, :, :, 5:7] = 0
        with patch.dict(loss._doris_balanced_loss.__globals__,
                        prob_to_edge=lambda pred, eps: edge_prediction):
            loss(logits, label, struct_edge_weight=reliability,
                 struct_edge_valid_mask=valid).backward()
        self.assertEqual(edge_prediction.grad[:, :, :, 5:7].abs().sum().item(), 0)
        # Real Sobel-on-probability path also backpropagates to model logits.
        loss(logits, label, struct_edge_weight=reliability).backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_loss_empty_classes_ignore_zero_weights_and_outer_weight(self):
        cls = edge_loss()['StructEdgeLoss']
        logits = torch.randn(1, 5, 12, 12, requires_grad=True)
        loss = cls(edge_radius=1, boundary_loss_mode='doris_balanced', ce_weight=0.1)
        for fill in (0, 255):
            label = torch.full((1, 12, 12), fill).long()
            value = loss(logits, label, struct_edge_weight=torch.ones(1, 1, 12, 12))
            self.assertTrue(torch.isfinite(value))
            self.assertEqual(value.item(), 0)
            value.backward()
        label = torch.zeros(1, 12, 12).long()
        label[:, :, 6:] = 1
        full = loss(logits, label, struct_edge_weight=torch.ones(1, 1, 12, 12))
        loss.loss_weight = 0.1
        scaled = loss(logits, label, struct_edge_weight=torch.ones(1, 1, 12, 12))
        torch.testing.assert_close(scaled, full * 0.1)
        zero = loss(logits, label, struct_edge_weight=torch.zeros(1, 1, 12, 12))
        self.assertEqual(zero.item(), 0)
        zero.backward()
        dense = torch.arange(144).reshape(1, 12, 12).remainder(2).long()
        self.assertTrue(torch.isfinite(loss(logits, dense)))  # no local negatives

    def test_classmix_mask_alignment_excludes_seam_and_invalid_target(self):
        model = new_model()
        mix = torch.zeros(1, 32, 32)
        mix[:, :, :16] = 1
        source = torch.ones(1, 32, 32).long()
        target_valid = torch.ones(1, 1, 32, 32).bool()
        target_valid[:, :, :3] = False
        result = model._doris_mix_valid_mask(mix, source, target_valid)
        self.assertEqual(result.shape, target_valid.shape)
        self.assertTrue(result[:, :, :3, :6].all())
        self.assertFalse(result[:, :, :3, 25:].any())
        self.assertFalse(result[:, :, :, 15:17].any())
        self.assertTrue(result[:, :, 8:, 25:].all())

    @unittest.skipUnless(HAS_BASELINE, 'Local stage-1 original snapshot unavailable')
    def test_legacy_structure_boundary_loss_values_gradients_states_and_dacs_weights(self):
        with patch.object(helpers, 'original_source', stage_source):
            old_cls = helpers.component('mmseg/models/uda/struct_prior_gate.py',
                                        'StructPriorConsistencyGate', original=True)
            old_dacs_cls = helpers.dacs_component(original=True)
        new_cls = helpers.component('mmseg/models/uda/struct_prior_gate.py',
                                    'StructPriorConsistencyGate')
        old_loss, new_loss = edge_loss(True)['StructEdgeLoss'], edge_loss()['StructEdgeLoss']
        label = (torch.arange(100).reshape(1, 10, 10) % 5).long()
        label[:, 0] = 255
        for step in (0, 7000, 12000):
            old, new = old_cls(num_classes=5), new_cls(num_classes=5)
            probs = torch.randn(1, 5, 10, 10).softmax(1)
            a, old_debug = old.apply(target_probs=probs, pseudo_label=label,
                                     source_gt=label, source_probs=probs,
                                     pseudo_weight=torch.ones_like(label).float(), iter=step)
            b, new_debug = new.apply(target_probs=probs, pseudo_label=label,
                                     source_gt=label, source_probs=probs,
                                     pseudo_weight=torch.ones_like(label).float(), iter=step)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key in old_debug:
                torch.testing.assert_close(old_debug[key], new_debug[key], rtol=0, atol=0)
            for key in old.state_dict():
                torch.testing.assert_close(old.state_dict()[key], new.state_dict()[key], rtol=0, atol=0)
            logits = torch.randn(1, 5, 10, 10, requires_grad=True)
            clone = logits.detach().clone().requires_grad_()
            weight = torch.rand(1, 1, 10, 10)
            a_loss, b_loss = old_loss()(logits, label, struct_edge_weight=weight), new_loss()(
                clone, label, struct_edge_weight=weight)
            a_loss.backward()
            b_loss.backward()
            torch.testing.assert_close(a_loss, b_loss, rtol=0, atol=0)
            torch.testing.assert_close(logits.grad, clone.grad, rtol=0, atol=0)
        cfg = helpers.static_config('configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py')
        kwargs = dict(cfg['uda'], model=cfg['model'], max_iters=20000)
        old, new = old_dacs_cls(**kwargs), helpers.dacs_component()(**kwargs)
        image = torch.randn(1, 6, 10, 10)
        for step in (0, 2500, 12000):
            weights = []
            for model in (old, new):
                model.local_iter = step
                weights.append(model._build_struct_edge_weight(image, label, 0.004, {},
                                                               'src', is_source=True))
                weights.append(model._build_struct_edge_weight(image, label, 0.002, {},
                    'tgt', target_probs=probs, is_target=True))
            torch.testing.assert_close(weights[0], weights[2], rtol=0, atol=0)
            torch.testing.assert_close(weights[1], weights[3], rtol=0, atol=0)
            torch.testing.assert_close(old.struct_edge_pair_prior, new.struct_edge_pair_prior,
                                       rtol=0, atol=0)


class DorisConfigTests(unittest.TestCase):
    def test_static_final_config_modules_parameters_and_baseline_parity(self):
        from tools.check_doris_stage1_configs import audit_configs, DIRECTIONS
        rows = audit_configs(helpers.static_config)
        self.assertEqual(len(rows), 30)
        parameters = ((3000, 6000, .0003, .95, .35, .006, .003),
                      (4000, 8000, .0002, .98, .35, .004, .002),
                      (3000, 6000, .001, .88, .70, .010, .006))
        for direction, values in zip(DIRECTIONS, parameters):
            stem = direction + '_hrda_3vi'
            old = helpers.static_config('configs/hrda/' + stem + '.py')
            baseline = helpers.static_config('configs/hrda/doris/' + direction + '/doris_' + stem + '.py')
            for flag in ('mechanism_logging', 'fixed_probe_enabled'):
                self.assertFalse(baseline['uda'].pop(flag))
            for key in ('model', 'uda', 'data', 'optimizer', 'optimizer_config', 'lr_config',
                        'checkpoint_config', 'evaluation', 'fp16', 'seed', 'crop_size'):
                self.assertEqual(old[key], baseline[key], (direction, key))
            self.assertEqual(old['runner']['max_iters'],20000)
            self.assertEqual(baseline['runner']['max_iters'],15000)
            full = helpers.static_config('configs/hrda/doris/' + direction + '/doris_' + stem + '_proto_struct_edge.py')
            proto, edge = full['uda']['proto_cfg'], full['uda']['struct_edge_cfg']
            self.assertEqual((proto['target_update_start'], proto['weight_start'],
                proto['lambda_proto_tgt'], proto['proto_min_weight'], proto['proto_power'],
                edge['loss_weight_src'], edge['loss_weight_tgt']), values)
            self.assertEqual(proto['loss_start'], proto['weight_start'])
            self.assertEqual(full['runner']['max_iters'],15000)
            for key in ('data', 'optimizer', 'optimizer_config', 'lr_config',
                        'checkpoint_config', 'evaluation', 'fp16', 'seed', 'crop_size'):
                self.assertEqual(full[key], old[key], (direction, key))

    @unittest.skipUnless(HAS_BASELINE, 'Local stage-1 original snapshot unavailable')
    def test_existing_configs_and_batch_scripts_are_byte_identical(self):
        snapshot = json.loads((ROOT / 'outputs/doris_stage1_before.json').read_text())
        for path, digest in snapshot['hashes'].items():
            if Path(path).name.startswith('doris_') or path.startswith('train_doris'):
                continue
            if path.startswith('configs/') or path.endswith('.sh'):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)

    def test_all_doris_experiment_configs_use_15000_iterations(self):
        paths=sorted((ROOT/'configs').rglob('doris_*.py'))
        self.assertEqual(len(paths),29)
        for path in paths:
            cfg=helpers.static_config(path.relative_to(ROOT).as_posix())
            self.assertEqual(cfg['runner']['max_iters'],15000,path.name)
            self.assertTrue(cfg['name_opt'].endswith('_15k'),path.name)

    @unittest.skipUnless(importlib.util.find_spec('mmcv'), 'MMCV unavailable locally')
    def test_real_mmcv_Config_fromfile_all_30(self):
        from mmcv import Config
        from tools.check_doris_stage1_configs import audit_configs
        self.assertEqual(len(audit_configs(Config.fromfile)), 30)


class DorisLayoutTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib unavailable locally')
    def test_response_strip_renderer_with_numpy_fixtures(self):
        """Test real plotting code separately when Torch/Matplotlib envs differ."""
        import numpy as np
        from PIL import Image
        tree = ast.parse((ROOT / 'tools/export_doris_response_maps.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'save_strip')
        scope = dict(np=np, copy=copy)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'layout', 'exec'), scope)
        class ArrayTensor:
            def __init__(self, value):
                self.value = value
            def __getitem__(self, key):
                return ArrayTensor(self.value[key])
            def detach(self):
                return self
            def float(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.value
        maps = {name: ArrayTensor(np.linspace(0, 1, 64).reshape(1, 8, 8)) for name in
                ('R_conf', 'R_continuity', 'R_adj', 'R_sem', 'R_struct',
                 'E_prob', 'E_img', 'R_pair', 'R_edge')}
        maps.update(adjacency_available=ArrayTensor(np.ones((1, 8, 8), dtype=bool)),
                    semantic_available=ArrayTensor(np.ones((1, 8, 8), dtype=bool)))
        directory = ROOT / 'outputs/doris_stage1_layout_test'
        directory.mkdir(exist_ok=True)
        for filename, names in (('mparc.png', ('R_conf', 'R_continuity', 'R_adj', 'R_sem', 'R_struct')),
                                ('pbcr.png', ('E_prob', 'E_img', 'R_pair', 'R_edge'))):
            output = directory / filename
            scope['save_strip'](maps, names, output)
            with Image.open(output) as image:
                self.assertEqual(image.height, 560)
                self.assertEqual(image.width, len(names) * 480)


if __name__ == '__main__':
    unittest.main(verbosity=2)
