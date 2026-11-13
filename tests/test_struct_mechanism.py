"""Structural replay acceptance with explicitly marked synthetic inputs."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_struct_mechanism_figure as figure


def fixture_arrays(ramp=1., mode='soft', positive_weight=.8, gt=False):
    y, x = np.mgrid[:256, :256]
    label = np.minimum(x//52, 4).astype(np.uint8)
    valid = np.zeros((256, 256), bool)
    valid[30:226, 30:226] = True
    ready_adj = np.isin(label, [2, 4])
    ready_sem = label == 2
    confidence = np.full((256, 256), .97, np.float32)
    continuity = (x/510).astype(np.float32)
    adjacency = np.where(ready_adj, .1+y/640, 1).astype(np.float32)
    semantic = np.where(ready_sem, .3, 1).astype(np.float32)
    response = confidence * (.4*continuity+.3*adjacency*ready_adj+.3*semantic*ready_sem) / (
        .4+.3*ready_adj+.3*ready_sem)
    response = response.astype(np.float32)
    scope = valid & np.isin(label, [2, 3, 4])
    gate = np.maximum(response, .65)
    gate = np.where(response < .32, 0 if mode == 'hard' else gate*.5, gate)
    gate = np.where(np.isin(label, [2, 3, 4]), (1-ramp)+ramp*gate, 1).astype(np.float32)
    before = valid.astype(np.float32)*positive_weight
    positive = scope & (before > 0)
    arrays = dict(image_rgb=np.stack([x, y, (x+y)//2], axis=-1).astype(np.uint8),
        ndvi=(x/127.5-1).astype(np.float32), ndre=(y/127.5-1).astype(np.float32),
        cire=(x/51).astype(np.float32), teacher_pseudo_label=label, teacher_confidence=confidence,
        R_conf=confidence.copy(), R_loc=continuity, R_adj=adjacency, R_prob=semantic,
        R_struct=response, G_struct=gate, A_struct=1-gate, W_before=before, W_after=before*gate,
        delta_W=before*gate-before, valid_mask=valid, adjacency_available=ready_adj,
        semantic_available=ready_sem, teacher_struct_mask=scope, positive_base_weight_mask=before > 0,
        positive_struct_weight_mask=positive, gate_active_mask=positive & (gate < 1),
        below_threshold_mask=scope & (response < .32), baseline_prediction=np.roll(label, 8, axis=1),
        struct_prediction=label.copy())
    if gt:
        arrays['gt_label'] = label.copy()
        arrays['gt_label'][60:100, 110:160] = 4
        arrays['gt_label'][:40] = 255
    meta = dict(direction='goji26RGB2goji30RGB', synthetic_fixture=True, sample_id='SYNTHETIC.tif',
        teacher_network='Struct EMA Teacher', prediction_network='Student',
        class_palette=figure.PALETTE.tolist(), class_names=['goji', 'vegetation', 'building', 'water', 'road'],
        pseudo_threshold=.968, checkpoint_struct=dict(completed_iter=12000, state_sha256_before='fixture'),
        config_struct=dict(resolved_sha256='fixture'), original_shape=[256, 256],
        gate=dict(fusion_mode='doris_confidence_guided', continuity_weight=.4, adjacency_weight=.3,
            semantic_weight=.3, struct_min_weight=.65, struct_filter_threshold=.32, struct_filter_mode=mode,
            struct_positive_classes=[2, 3, 4], ramp=ramp, use_target_gate=True, eps=1e-6,
            struct_warmup_iters=6000, struct_rampup_iters=4000,
            rule_iter=int(6000+4000*ramp), state='active'),
        vi_normalization=dict(ndvi=dict(min=-1., max=1.), ndre=dict(min=-1., max=1.), cire=dict(min=0., max=5.)))
    return arrays, meta


class StructuralFigureTests(unittest.TestCase):
    def test_formula_threshold_schedule_and_empty_positive_weight(self):
        for ramp in (0., .5, 1.):
            for mode in ('soft', 'hard'):
                for weight in (0., .8):
                    arrays, meta = fixture_arrays(ramp, mode, weight)
                    figure.validate_arrays(arrays, meta)
                    np.testing.assert_array_equal(figure.crop_arrays(arrays)['R_struct'], arrays['R_struct'][30:226, 30:226])
                    stats = figure.descriptive_statistics(arrays, meta)
                    self.assertEqual(stats['gate_active_pixels'], int(arrays['gate_active_mask'].sum()))
                    self.assertEqual(stats['gt_available'], False)
                    if weight == 0:
                        self.assertIsNone(stats['mean_attenuation'])
                        self.assertTrue(np.any(arrays['G_struct'] < 1) if ramp else np.all(arrays['G_struct'] == 1))

    def test_bad_identity_masks_network_and_nonfinite_rejected(self):
        arrays, meta = fixture_arrays()
        for key in ('R_struct', 'G_struct', 'A_struct', 'W_after', 'delta_W'):
            changed = copy.deepcopy(arrays)
            changed[key][80, 130] += .05
            with self.assertRaises(ValueError, msg=key):
                figure.validate_arrays(changed, meta)
        changed = copy.deepcopy(arrays)
        changed['adjacency_available'][80, 130] = False
        with self.assertRaises(ValueError):
            figure.validate_arrays(changed, meta)
        with self.assertRaises(ValueError):
            figure.validate_arrays(arrays, dict(meta, prediction_network='EMA Teacher'))
        arrays['R_loc'][80, 130] = np.nan
        with self.assertRaises(ValueError):
            figure.validate_arrays(arrays, meta)

    def test_missing_priors_and_zero_weight_are_gray_and_excluded(self):
        arrays, meta = fixture_arrays(positive_weight=0.)
        for key in ('R_adj', 'R_prob', 'A_struct'):
            mask = figure.applicability(arrays, key)
            pixels = figure.colored_pixels(arrays, key, meta)
            np.testing.assert_array_equal(pixels[~mask], np.broadcast_to(figure.GRAY, pixels[~mask].shape))
        stats = figure.descriptive_statistics(arrays, meta)
        row = next(r for r in stats['rows'] if r['group'] == 'teacher_water' and r['metric'] == 'R_adj')
        self.assertEqual(row['count'], 0)
        self.assertIsNone(row['mean'])
        arrays['gt_label'] = np.full((256, 256), 255, np.uint8)
        stats = figure.descriptive_statistics(arrays, meta)
        self.assertIsNone(stats['error_detection']['A_struct']['roc_auc'])

    def test_amp_response_roundoff_preserves_strict_gate_and_weight_checks(self):
        arrays, meta = fixture_arrays()
        arrays['R_struct'][80, 130] += .0005
        with self.assertRaisesRegex(ValueError, 'fusion identity'):
            figure.validate_arrays(arrays, meta)
        meta['checkpoint_struct']['struct_probability_dtype'] = 'float16'
        figure.validate_arrays(arrays, meta)
        for key in ('G_struct', 'W_after'):
            changed = copy.deepcopy(arrays)
            changed[key][80, 130] += .0001
            with self.assertRaisesRegex(ValueError, 'identity'):
                figure.validate_arrays(changed, meta)

    def test_geometry_and_titles_fit_for_both_plates(self):
        arrays, meta = fixture_arrays()
        for keys in (figure.COMPONENTS, figure.COMPARISON):
            fig = figure.group_figure(arrays, meta, keys, roi=[80, 80, 145, 145])
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                axes = [a for a in fig.axes if a.images]
                self.assertEqual(len(axes), len(keys))
                width, height = fig.get_size_inches()
                legend = fig.legends[0].get_window_extent(renderer) if fig.legends else None
                for axis in axes:
                    box = axis.get_position()
                    self.assertAlmostEqual(box.width*width, box.height*height)
                    title = axis.title.get_window_extent(renderer)
                    self.assertLessEqual(title.y1, fig.bbox.y1)
                    self.assertGreaterEqual(title.x0, 0)
                    self.assertLessEqual(title.x1, fig.bbox.x1)
                    if legend is not None:
                        self.assertFalse(legend.overlaps(axis.get_tightbbox(renderer)))
                self.assertEqual(axes[0].images[0].get_interpolation(), 'nearest')
            finally:
                figure.plt.close(fig)

    def test_all_exports_native_pixels_immutable_inputs_and_collision(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            root = Path(temporary)
            arrays, meta = fixture_arrays(gt=True)
            full = root / 'input.npz'
            np.savez_compressed(full, **arrays)
            meta['arrays_sha256'] = {'responses_full_256.npz': figure.file_sha256(full)}
            metadata = root / 'input.json'
            metadata.write_text(json.dumps(meta), encoding='utf-8')
            output = root / 'render'
            rendered = figure.render(full, metadata, output, roi=[80, 80, 145, 145], dpi=45)
            for name in ('struct_full_raw.png', 'struct_full_raw.pdf',
                         'struct_formal_group_1x7.png', 'struct_formal_group_1x7.pdf',
                         'target_rgb_formal.png', 'r_conf_struct_only_formal.png',
                         'r_loc_struct_only_formal.png', 'r_adj_struct_only_formal.png',
                         'r_prob_struct_only_formal.png', 'r_struct_struct_only_formal.png',
                         'struct_student_prediction_formal.png',
                         'struct_only_raw.png', 'struct_only_contrast.png', 'struct_contrib.png',
                         'struct_comparison_1x5.png', 'struct_comparison_1x5.pdf',
                         'struct_stats.json', 'struct_stats.csv', 'render_meta.json',
                         'responses_center_196.npz', 'panels_196/A_struct_colorbar.png',
                         'panels_196/baseline_vs_struct_roi.png', 'logs/render.log'):
                self.assertTrue((rendered / name).is_file(), name)
            from tools.doris_struct_evidence import prepare_struct_evidence
            from tools.struct_mechanism_panels import display_scales, pixels_for
            prepared, prepared_meta = prepare_struct_evidence(arrays, meta)
            scales = display_scales(figure.crop_arrays(prepared), prepared_meta, 'struct')
            for size in (196, 256):
                current = figure.crop_arrays(prepared) if size == 196 else prepared
                for key in ('R_adj', 'A_struct'):
                    with Image.open(rendered / ('panels_'+str(size)) / 'struct_only' / (key+'_no_title.png')) as png:
                        self.assertEqual(png.size, (size, size))
                        np.testing.assert_array_equal(np.asarray(png.convert('RGB')), pixels_for(current, key, 'struct', scales))
            self.assertEqual(figure.file_sha256(full), meta['arrays_sha256']['responses_full_256.npz'])
            with self.assertRaises(FileExistsError):
                figure.render(full, metadata, output, dpi=46)
            figure.export_online_views(full, metadata, output, dpi=45)
            for name in ('selected_arrays/iter_012000.npz',
                         'snapshots/struct_v4/iter_012000/struct_formal_group_1x7.png',
                         'snapshots/struct_v4/iter_012000/panels_196/A_struct_raw.pdf',
                         'snapshots/struct_v4/iter_012000/panels_196/struct_prediction_no_title.png'):
                self.assertTrue((output / 'fixed_probe' / name).is_file(), name)
            stats = json.loads((rendered / 'struct_stats.json').read_text())
            self.assertTrue(stats['gt_available'])
            self.assertGreater(stats['teacher_errors'], 0)


HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ('torch', 'mmcv'))


@unittest.skipUnless(HAS_RUNTIME, 'Torch/MMCV are required for production forward verification')
class StructuralForwardTests(unittest.TestCase):
    def test_production_gate_persisted_priors_readonly_forward_and_neutral_warmup(self):
        import torch
        from tools import replay_struct_mechanism as replay
        from mmseg.models.uda.struct_prior_gate import StructPriorConsistencyGate
        from mmseg.utils.mechanism import probe_arrays

        class TinyStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Conv2d(6, 5, 1)

            def generate_pseudo_label(self, image, metas):
                return self.layer(image)

            def forward(self, img, img_metas, return_loss=False, rescale=True):
                return [self.layer(img[0]).argmax(1)[0].cpu().numpy()]

        class TinyDACS(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model, self.ema_model = TinyStudent(), TinyStudent()
                self.struct_gate = StructPriorConsistencyGate(5, fusion_mode='doris_confidence_guided')
                self.local_iter, self.num_classes, self.pseudo_threshold = 12000, 5, 0.
                self.enable_struct = True
                self.enable_phy = self.enable_proto = self.enable_struct_edge = False

            def get_model(self): return self.model
            def get_ema_model(self): return self.ema_model
            def _doris_target_valid_mask(self, label, spatial): return spatial.bool()

        torch.manual_seed(3)
        original = TinyDACS()
        truth = torch.randint(0, 5, (1, 1, 256, 256))
        original.struct_gate.update_source_adjacency(truth)
        original.struct_gate.update_semantic_prototype(torch.rand(1, 5, 256, 256).softmax(1), truth)
        model = TinyDACS()
        model.load_state_dict(original.state_dict(), strict=True)
        sample = (torch.rand(1, 6, 256, 256), torch.rand(1, 3, 256, 256), [{}])
        probe = dict(margins=[30, 30, 30, 30])
        before = replay.shared.state_digest(model.state_dict())
        arrays, digest = replay.infer_model(model, sample, probe)
        self.assertEqual(before, digest)
        self.assertEqual(before, replay.shared.state_digest(model.state_dict()))
        spatial = torch.zeros(1, 1, 256, 256)
        spatial[:, :, 30:226, 30:226] = 1
        online = probe_arrays(model, *sample, spatial)
        for saved, live in (('R_loc', 'r_loc'), ('R_adj', 'r_adj'), ('R_prob', 'r_prob'),
                            ('R_struct', 'r_struct'), ('W_before', 'w_before_struct'), ('W_after', 'w_after_struct')):
            np.testing.assert_array_equal(arrays[saved], online[live])
        self.assertTrue(np.any(arrays['G_struct'] < 1))
        model.local_iter = 2000
        warmup, _ = replay.infer_model(model, sample, probe)
        np.testing.assert_array_equal(warmup['G_struct'], 1)
        np.testing.assert_array_equal(warmup['W_before'], warmup['W_after'])


class StructuralLauncherTests(unittest.TestCase):
    def test_rgb_tableiv_structural_v2_alignment_and_output_roots(self):
        from test_minimal_output import static_config
        from tools.check_doris_stage1_configs import VARIANTS
        from tools.check_doris_stage1_ablation_configs import STRUCT_V2_OVERRIDES
        from tools.struct_schedule import struct_gate_strength

        folder = 'configs/hrda/doris/goji26RGB2goji30RGB/'
        stem = 'doris_goji26RGB2goji30RGB_hrda_3vi'
        configs = {suffix: static_config(folder+stem+suffix+'.py') for suffix, _ in VARIANTS}
        reference = configs['_struct_gate']['uda']['struct_cfg']
        for suffix in ('_struct_gate', '_phy_struct_gate', '_proto_struct_edge'):
            cfg = configs[suffix]
            self.assertEqual(cfg['uda']['struct_cfg'], reference)
            self.assertEqual(cfg['model']['decode_head']['struct_prior_interaction']['gamma_struct'], .05)
            for key, value in STRUCT_V2_OVERRIDES.items():
                self.assertEqual(reference[key], value)
            for step, expected in zip((0, 1000, 2000, 3000, 4000, 6000, 12000), (.1, .325, .55, .775, 1., 1., 1.)):
                self.assertAlmostEqual(struct_gate_strength(step, reference['struct_gate_schedule'],
                    initial_strength=reference['struct_gate_initial_strength'], full_iter=reference['struct_gate_full_iter']), expected)
        for suffix, flags in VARIANTS:
            cfg = configs[suffix]
            self.assertEqual(tuple(cfg['uda'][module+'_cfg']['enable'] for module in ('phy', 'struct', 'proto', 'struct_edge')), flags)
            self.assertEqual(cfg['work_dir'], '/root/shared-nvme/work_dirs/doris_struct_distinct_v2/'
                'goji26RGB2goji30RGB/'+stem[6:]+suffix)
            for key in ('data', 'optimizer', 'lr_config', 'runner', 'evaluation', 'checkpoint_config'):
                self.assertEqual(cfg[key], configs[''][key])
        for module, suffix in (('phy', '_phy_gate'), ('proto', '_ps_proto'), ('struct_edge', '_struct_edge')):
            self.assertEqual(configs['_proto_struct_edge']['uda'][module+'_cfg'], configs[suffix]['uda'][module+'_cfg'])
        self.assertEqual(configs['_phy_struct_gate']['uda']['phy_cfg'], configs['_phy_gate']['uda']['phy_cfg'])

    def test_resolved_configs_pass_shared_protocol_and_match_training_data(self):
        from test_minimal_output import static_config

        class Config(dict):
            __setattr__ = dict.__setitem__

            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as error:
                    raise AttributeError(name) from error

        def attributes(value):
            if isinstance(value, dict):
                return Config({key: attributes(child) for key, child in value.items()})
            if isinstance(value, list):
                return [attributes(child) for child in value]
            return value

        source = ROOT / 'tools/replay_hphy_mechanism.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                     node.name in ('verify_protocol', 'data_protocol')]
        scope = dict(Path=Path, deepcopy=copy.deepcopy)
        exec(compile(tree, str(source), 'exec'), scope)
        folder = 'configs/hrda/doris/goji26RGB2goji30RGB/'
        name = 'doris_goji26RGB2goji30RGB_hrda_3vi'
        structural = attributes(static_config(folder+name+'_struct_gate.py'))
        baseline = attributes(static_config(folder+name+'.py'))
        from tools.check_doris_stage1_ablation_configs import audit
        self.assertEqual(len(audit(lambda path: attributes(static_config(Path(path).relative_to(ROOT).as_posix())))), 24)
        scope['verify_protocol'](structural, False, structural=True)
        scope['verify_protocol'](baseline, False)
        self.assertEqual(scope['data_protocol'](structural), scope['data_protocol'](baseline))
        for key in ('seed', 'optimizer', 'lr_config', 'fp16', 'target_img_norm_cfg'):
            self.assertEqual(structural.get(key), baseline.get(key))
        self.assertEqual(structural.model.decode_head.struct_prior_interaction.gamma_struct, .05)
        self.assertEqual(structural.runner.max_iters, 15000)
        self.assertEqual(structural.uda.struct_cfg.struct_min_weight, .65)
        self.assertEqual(structural.uda.struct_cfg.struct_filter_threshold, .32)
        self.assertEqual(structural.uda.struct_cfg.struct_filter_mode, 'soft')
        # Execute the real CLI work-dir selection statements without importing MMCV.
        source = ROOT/'tools/train.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        selection_test = ast.dump(ast.parse('args.work_dir is not None', mode='eval').body)
        selection = next(node for node in main.body if isinstance(node, ast.If) and
                         ast.dump(node.test) == selection_test)
        assignment_target = ast.dump(ast.parse('cfg.model.train_cfg.work_dir = None').body[0].targets[0])
        assignment = next(node for node in main.body if isinstance(node, ast.Assign) and
                          ast.dump(node.targets[0]) == assignment_target)
        from types import SimpleNamespace
        tree.body = [selection, assignment]
        namespace = dict(cfg=structural, args=SimpleNamespace(work_dir='/tmp/struct-v2-cli'))
        exec(compile(tree, str(source), 'exec'), namespace)
        self.assertEqual(structural.work_dir, '/tmp/struct-v2-cli')
        self.assertEqual(structural.model.train_cfg.work_dir, '/tmp/struct-v2-cli')
        from tools.doris_checkpoint_plan import checkpoint_plan
        events = checkpoint_plan(structural)['events']
        self.assertTrue(events)
        self.assertTrue(all(event['local_iter'] == 4000 for event in events))
        with self.assertRaises(ValueError):
            scope['verify_protocol'](structural, True)

    def test_short_command_and_training_work_directory(self):
        tree = ast.parse((ROOT / 'tools/replay_struct_mechanism.py').read_text(encoding='utf-8'))
        defaults = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        path = ast.literal_eval(defaults['STRUCT_WORK_DIR'].args[0])
        self.assertEqual(path, '/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_gate')
        for key in ('CHECKPOINT_STRUCT', 'CHECKPOINT_BASELINE'):
            self.assertEqual(ast.literal_eval(defaults[key].args[0]), 'iter_12000.pth')
        phy = (ROOT / 'scripts/run_hphy_mechanism_26rgb2_30rgb.sh').read_text()
        expected = phy.replace('replay_hphy_mechanism', 'replay_struct_mechanism').replace('H-Phy', 'Struct')
        self.assertEqual((ROOT / 'scripts/run_struct_mechanism_26rgb2_30rgb.sh').read_text(), expected)

    @unittest.skipUnless(shutil.which('bash'), 'Bash is required for launcher execution')
    def test_launcher_failure_propagates_and_arguments_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            root = Path(temporary)
            (root / 'scripts').mkdir()
            (root / 'bin').mkdir()
            script = root / 'scripts/run.sh'
            script.write_bytes((ROOT / 'scripts/run_struct_mechanism_26rgb2_30rgb.sh').read_bytes())
            (root / 'bin/python').write_bytes(b'#!/usr/bin/env bash\nexit 31\n')
            invocation = 'cd -- "$1"; export PATH="$PWD/bin:$PATH"; chmod +x bin/python; bash scripts/run.sh "${@:2}"'
            for arguments, expected in (([], 31), (['extra'], 2)):
                result = subprocess.run([shutil.which('bash'), '-c', invocation, 'fixture', str(root), *arguments],
                                        capture_output=True, text=True, env=dict(os.environ))
                self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == '__main__':
    unittest.main()
