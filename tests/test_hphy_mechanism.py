"""Synthetic, explicitly marked offline-mechanism acceptance fixtures."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from tools import build_hphy_mechanism_figure as figure


def fixture_arrays(gt=False):
    y, x = np.mgrid[:256, :256]
    label = ((x // 52) % 5).astype(np.uint8)
    valid = np.zeros((256, 256), bool)
    valid[30:226, 30:226] = True
    response = (.5 + .4*x/255).astype(np.float32)
    conditional = (.1 + .8*y/255).astype(np.float32)
    p_goji, p_veg = response*conditional, response*(1-conditional)
    agreement = np.where(label == 0, p_goji, np.where(label == 1, p_veg, 1)).astype(np.float32)
    scope = valid & np.isin(label, [0, 1])
    gate = np.where(scope, 1-.25*(1-agreement), 1).astype(np.float32)
    before = valid.astype(np.float32)*.8
    arrays = dict(image_rgb=np.stack([x, y, (x+y)//2], axis=-1).astype(np.uint8),
        ndvi=(x/127.5-1).astype(np.float32), ndre=(y/127.5-1).astype(np.float32),
        cire=(5*x/255).astype(np.float32), teacher_pseudo_label=label,
        teacher_confidence=np.full((256, 256), .99, np.float32),
        baseline_prediction=np.roll(label, 20, axis=1), hphy_prediction=label.copy(),
        R_veg=response, Q_G=conditional, P_G=p_goji, P_V=p_veg, P_N=1-response,
        S_phy=agreement, G_phy=gate, A_phy=1-gate, W_before=before, W_after=before*gate,
        valid_mask=valid, teacher_goji_mask=valid & (label == 0),
        teacher_vegetation_mask=valid & (label == 1), teacher_phy_mask=scope)
    if gt:
        truth = label.copy()
        truth[60:100, 35:95] = 1-truth[60:100, 35:95]
        truth[:40] = 255
        arrays['gt_label'] = truth
    meta = dict(direction='goji26RGB2goji30RGB', synthetic_fixture=True,
        teacher_network='H-Phy EMA Teacher', prediction_network='Student',
        class_palette=figure.PALETTE.tolist(), pseudo_threshold=.968,
        gate=dict(min_weight=.75), vi_normalization=dict(
            ndvi=dict(min=-1., max=1.), ndre=dict(min=-1., max=1.), cire=dict(min=0., max=5.)))
    return arrays, meta


def write_fixture(folder, gt=False):
    folder = Path(folder)
    folder.mkdir(parents=True)
    arrays, meta = fixture_arrays(gt)
    path = folder / 'responses_full_256.npz'
    np.savez_compressed(path, **arrays)
    meta['arrays_sha256'] = {path.name: figure.file_sha256(path)}
    metadata = folder / 'mechanism_meta.json'
    metadata.write_text(json.dumps(meta), encoding='utf-8')
    return path, metadata, arrays, meta


class FigureTests(unittest.TestCase):
    def test_conflicting_inference_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            root = Path(temporary)
            path, metadata, _, _ = write_fixture(root / 'input')
            destination = root / 'output'
            destination.mkdir()
            existing = destination / 'mechanism_meta.json'
            existing.write_bytes(b'different provenance')
            with self.assertRaises(FileExistsError):
                figure.render(path, metadata, destination)
            self.assertEqual(existing.read_bytes(), b'different provenance')

    def test_exact_crop_scope_palette_and_roi(self):
        arrays, meta = fixture_arrays()
        figure.validate_arrays(arrays, meta)
        cropped = figure.crop_arrays(arrays)
        for key in arrays:
            np.testing.assert_array_equal(cropped[key], arrays[key][30:226, 30:226])
        self.assertEqual(figure.roi_coordinates([80, 80, 145, 145]), [50, 50, 115, 115])
        for roi in ([0, 30, 100, 100], [30, 30, 227, 100], [90, 90, 80, 100]):
            with self.assertRaises(ValueError):
                figure.roi_coordinates(roi)
        colored = figure.colored_pixels(arrays, 'S_phy', meta)
        np.testing.assert_array_equal(colored[~arrays['teacher_phy_mask']],
                                      np.broadcast_to(figure.GRAY, colored[~arrays['teacher_phy_mask']].shape))
        categorical = figure.colored_pixels(arrays, 'teacher_pseudo_label', meta)
        for cls, color in enumerate(figure.PALETTE):
            np.testing.assert_array_equal(categorical[arrays['teacher_pseudo_label'] == cls],
                                          np.broadcast_to(color, categorical[arrays['teacher_pseudo_label'] == cls].shape))

    def test_bad_numerics_identities_and_prediction_network_rejected(self):
        arrays, meta = fixture_arrays()
        for key in ('P_G', 'A_phy', 'W_after', 'S_phy'):
            modified = copy.deepcopy(arrays)
            modified[key][80, 40] += .1
            with self.assertRaises(ValueError):
                figure.validate_arrays(modified, meta)
        modified = dict(meta, prediction_network='EMA Teacher')
        with self.assertRaisesRegex(ValueError, 'identities'):
            figure.validate_arrays(arrays, modified)
        with self.assertRaisesRegex(ValueError, 'doris_stage1'):
            figure.output_path(ROOT / 'outputs/doris_stage1/wrong')

    def test_amp_probability_roundoff_keeps_gate_and_weight_checks_strict(self):
        arrays, meta = fixture_arrays()
        vegetation = arrays['R_veg'].astype(np.float16)
        conditional = arrays['Q_G'].astype(np.float16)
        arrays.update(R_veg=vegetation.astype(np.float32), Q_G=conditional.astype(np.float32),
            P_G=(vegetation*conditional).astype(np.float32),
            P_V=(vegetation*(1-conditional)).astype(np.float32),
            P_N=(1-vegetation).astype(np.float32))
        label = arrays['teacher_pseudo_label']
        arrays['S_phy'] = np.where(label == 0, arrays['P_G'],
                                  np.where(label == 1, arrays['P_V'], 1)).astype(np.float32)
        arrays['G_phy'] = np.where(arrays['teacher_phy_mask'],
                                   1-.25*(1-arrays['S_phy']), 1).astype(np.float32)
        arrays['A_phy'] = 1-arrays['G_phy']
        arrays['W_after'] = arrays['W_before']*arrays['G_phy']
        with self.assertRaisesRegex(ValueError, 'identity'):
            figure.validate_arrays(arrays, meta)
        meta['checkpoint_phy'] = dict(hphy_probability_dtype='float16')
        figure.validate_arrays(arrays, meta)
        for key, change in (('P_G', .01), ('A_phy', .0001), ('W_after', .0001)):
            invalid = copy.deepcopy(arrays)
            invalid[key][80, 40] += change
            with self.assertRaisesRegex(ValueError, 'identity'):
                figure.validate_arrays(invalid, meta)

    def test_group_geometry_matching_roi_and_no_gt_statistics(self):
        arrays, meta = fixture_arrays()
        self.assertFalse(figure.descriptive_statistics(arrays, meta)['available'])
        fig = figure.group_figure(arrays, meta, [80, 80, 145, 145])
        try:
            axes = [axis for axis in fig.axes if axis.images]
            self.assertEqual(len(axes), 7)  # Five columns and two matched ROI images.
            np.testing.assert_array_equal(axes[1].images[0].get_array(),
                                          figure.colored_pixels(figure.crop_arrays(arrays), 'teacher_confidence', meta))
            bars = [axis for axis in fig.axes if not axis.images]
            self.assertEqual(len(bars), 3)
            for axis, ticks in zip(bars, ([0, .5, 1], [0, .5, 1], [0, .125, .25])):
                np.testing.assert_allclose(axis.get_xticks(), ticks)
            boxes = [axis.get_position() for axis in axes[:5]]
            width, height = fig.get_size_inches()
            for box in boxes:
                self.assertAlmostEqual(box.width * width, box.height * height)
                self.assertAlmostEqual(box.y0, boxes[0].y0)
            for axis in axes:
                self.assertEqual(axis.images[0].get_interpolation(), 'nearest')
            for axis, key in zip(axes[-2:], ('baseline_prediction', 'hphy_prediction')):
                np.testing.assert_array_equal(axis.images[0].get_array(),
                    figure.colored_pixels(arrays, key, meta)[80:145, 80:145])
        finally:
            figure.plt.close(fig)

    def test_gt_groups_ignore_empty_and_safe_auc(self):
        arrays, meta = fixture_arrays(gt=True)
        stats = figure.descriptive_statistics(arrays, meta)
        for row in stats['groups']:
            truth, label = arrays['gt_label'][30:226, 30:226], arrays['teacher_pseudo_label'][30:226, 30:226]
            expected = (truth == row['gt']) & (label == row['teacher'])
            self.assertEqual(row['pixels'], int(expected.sum()))
        self.assertEqual(stats['scope'], 'center196')
        arrays['gt_label'][:] = 255
        empty = figure.descriptive_statistics(arrays, meta)
        self.assertTrue(all(row['pixels'] == 0 and row['S_phy'] is None for row in empty['groups']))
        self.assertIsNone(empty['auroc']['A_phy'])
        self.assertIsNotNone(empty['auroc_null_reason'])

    def test_group_titles_colorbars_and_legend_do_not_overlap(self):
        arrays, meta = fixture_arrays()
        for roi in (None, [80, 80, 145, 145]):
            fig = figure.group_figure(arrays, meta, roi)
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                legend = fig.legends[0].get_window_extent(renderer)
                for axis in fig.axes:
                    box = axis.get_window_extent(renderer) if axis.images else axis.get_tightbbox(renderer)
                    self.assertFalse(legend.overlaps(box))
                    self.assertLessEqual(axis.title.get_window_extent(renderer).y1, fig.bbox.y1)
            finally:
                figure.plt.close(fig)

    def test_all_exports_pixel_matching_and_collision_without_gt(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            root = Path(temporary)
            path, metadata, arrays, meta = write_fixture(root / 'input')
            destination = root / 'render'
            figure.render(path, metadata, destination, roi=[80, 80, 145, 145], dpi=80)
            for name in ('phy_mechanism_1x5.png', 'phy_mechanism_1x5.pdf',
                         'responses_full_256.npz', 'responses_center_196.npz', 'phy_stats.json', 'phy_stats.csv',
                         'colorbars/s_phy.png', 'colorbars/attenuation.png', 'colorbars/confidence.png',
                         'panels_196/baseline_vs_hphy_roi.png'):
                self.assertTrue((destination / name).is_file(), name)
            self.assertTrue((destination / 'phy_mechanism_1x5.pdf').read_bytes().startswith(b'%PDF'))
            for size in (196, 256):
                data = figure.crop_arrays(arrays) if size == 196 else arrays
                folder = destination / ('panels_' + str(size))
                for name, key, _ in figure.MAIN:
                    for prefix in (folder, folder / 'clean'):
                        with Image.open(prefix / (name + '.png')) as png:
                            self.assertEqual(png.size, (size, size))
                            np.testing.assert_array_equal(np.asarray(png.convert('RGB')),
                                                          figure.colored_pixels(data, key, meta))
                    with Image.open(folder / 'labeled' / (name + '.png')) as png:
                        png.verify()
            with np.load(destination / 'responses_center_196.npz') as cropped:
                for key in arrays:
                    np.testing.assert_array_equal(cropped[key], arrays[key][30:226, 30:226])
            settings = json.loads((destination / 'render_meta.json').read_text())
            self.assertEqual(settings['render']['attenuation_range'], [0, .25])
            self.assertEqual(settings['render']['roi_center_196'], [50, 50, 115, 115])
            self.assertFalse(json.loads((destination / 'phy_stats.json').read_text())['available'])
            hashes = {p: figure.file_sha256(p) for p in destination.rglob('*') if p.is_file()}
            with self.assertRaises(FileExistsError):
                figure.render(path, metadata, destination, dpi=80)
            self.assertEqual(hashes, {p: figure.file_sha256(p) for p in hashes})

    def test_render_only_cli_imports_no_torch_and_no_roi_is_supported(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            root = Path(temporary)
            path, metadata, _, _ = write_fixture(root / 'input')
            code = ('import sys; from tools.build_hphy_mechanism_figure import render; '
                    'assert "torch" not in sys.modules; render(sys.argv[1],sys.argv[2],sys.argv[3],'
                    'dpi=60,export_individual=False,export_full=False); assert "torch" not in sys.modules')
            subprocess.run([sys.executable, '-c', code, str(path), str(metadata), str(root / 'output')],
                           cwd=ROOT, check=True, capture_output=True)
            self.assertFalse((root / 'output/panels_196/baseline_vs_hphy_roi.png').exists())


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Torch is required for synthetic checkpoint replay')
class PairedReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mmcv import Config
        from mmseg.models.uda.dacs import DACS
        from mmseg.models.uda.phy_reliability import HierarchicalPhyReliabilityHead, phy_gate_parameters
        from tools import replay_hphy_mechanism
        cls.api, cls.Config = replay_hphy_mechanism, Config
        cls.calls = []
        torch.set_num_threads(2)

        class SmallSegmentation(torch.nn.Module):
            def __init__(self, role, prediction):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(6)
                self.classifier = torch.nn.Conv2d(6, 5, 1)
                self.role = role
                with torch.no_grad():
                    self.classifier.weight.zero_()
                    self.classifier.bias.fill_(-8)
                    self.classifier.bias[prediction] = 8

            def generate_pseudo_label(self, image, metas):
                cls.calls.append((self.role, 'teacher', self.training, torch.is_grad_enabled(),
                                  torch.is_inference_mode_enabled(), tuple(image.shape)))
                return self.classifier(self.bn(image))

            def simple_test(self, image, metas, rescale=True):
                cls.calls.append((self.role, 'student', self.training, torch.is_grad_enabled(),
                                  torch.is_inference_mode_enabled(), tuple(image.shape)))
                return [self.classifier(self.bn(image)).argmax(1)[0].cpu().numpy()]

            def forward(self, img, img_metas, return_loss=False, rescale=True):
                if return_loss:
                    raise AssertionError('Training is forbidden in offline replay')
                return self.simple_test(img[0], img_metas[0], rescale=rescale)

        class SmallDACS(torch.nn.Module):
            _apply_phy_gate = DACS._apply_phy_gate
            _doris_target_valid_mask = DACS._doris_target_valid_mask
            _class_mask = DACS._class_mask

            def __init__(self, config):
                super().__init__()
                self.enable_phy = config.uda.phy_cfg.enable
                self.num_classes = 5
                self.psweight_ignore_top = self.psweight_ignore_bottom = 0
                self.pseudo_threshold = config.uda.pseudo_threshold
                self.model = SmallSegmentation('hphy' if self.enable_phy else 'baseline', 1 if self.enable_phy else 2)
                self.ema_model = SmallSegmentation('hphy' if self.enable_phy else 'baseline', 0)
                if self.enable_phy:
                    self.phy_apply_to_classes = config.uda.phy_cfg.get('apply_to_classes', [0, 1])
                    self.phy_head = HierarchicalPhyReliabilityHead()
                    self.phy_mode = 'hierarchical_gv'
                    self.phy_gate_params = phy_gate_parameters(config.uda.phy_cfg)
                    self.phy_min_weight = config.uda.phy_cfg.phy_min_weight
                    self.phy_warmup_iters = config.uda.phy_cfg.phy_warmup_iters
                    self.phy_rampup_iters = config.uda.phy_cfg.phy_rampup_iters
                    self.use_target_gate = True

            def get_model(self): return self.model
            def get_ema_model(self): return self.ema_model

        cls.model_class = SmallDACS

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / 'outputs')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.target = self.root / '30/rgb_dir/val/SYNTHETIC.tif'
        self.target.parent.mkdir(parents=True)
        image = fixture_arrays()[0]['image_rgb']
        Image.fromarray(image).save(self.target)
        for name in ('ndvi', 'ndre', 'cire'):
            folder = self.root / ('30/' + name + '_dir/val')
            folder.mkdir(parents=True)
            Image.fromarray(fixture_arrays()[0][name]).save(folder / self.target.name)
        self.phy_path = ROOT / 'configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py'
        self.base_path = ROOT / 'configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py'
        self.phy_cfg = self.Config.fromfile(str(self.phy_path))
        self.base_cfg = self.Config.fromfile(str(self.base_path))
        self.phy_checkpoint = self.root / 'hphy_best_mIoU_iter_12000.pth'
        self.base_checkpoint = self.root / 'baseline_best_goji_mIoU_iter_15000.pth'
        for config, destination, iteration in ((self.phy_cfg, self.phy_checkpoint, 12000),
                                               (self.base_cfg, self.base_checkpoint, 15000)):
            model = self.model_class(config)
            torch.save(dict(state_dict=model.state_dict(), meta=dict(iter=iteration-1,
                       config=config.pretty_text, synthetic_fixture=True)), destination)
        self.calls.clear()

    def replay(self, folder='replay', **kwargs):
        with patch('mmseg.models.builder.build_train_model', side_effect=self.model_class):
            return self.api.replay(self.phy_path, self.phy_checkpoint, self.base_path,
                self.base_checkpoint, self.target, self.root / folder, device='cpu', **kwargs)

    def test_two_checkpoints_real_loader_whole_grid_identity_and_no_gt(self):
        hashes = [figure.file_sha256(p) for p in (self.phy_checkpoint, self.base_checkpoint)]
        out = self.replay()
        meta = json.loads((out / 'mechanism_meta.json').read_text())
        self.assertEqual(meta['checkpoint_phy']['completed_iter'], 12000)
        self.assertEqual(meta['checkpoint_baseline']['completed_iter'], 15000)
        self.assertTrue(meta['synthetic_fixture'])
        self.assertIsNone(meta['gt'])
        self.assertEqual([(row[0], row[1]) for row in self.calls],
                         [('hphy', 'teacher'), ('hphy', 'student'), ('baseline', 'student')])
        self.assertTrue(all(not row[2] and not row[3] and row[4] and row[5] == (1, 6, 256, 256) for row in self.calls))
        with np.load(out / 'responses_full_256.npz') as full, np.load(out / 'responses_center_196.npz') as center:
            self.assertTrue((full['teacher_pseudo_label'] == 0).all())
            self.assertTrue((full['hphy_prediction'] == 1).all())
            self.assertTrue((full['baseline_prediction'] == 2).all())
            for key in full.files:
                np.testing.assert_array_equal(center[key], full[key][30:226, 30:226])
        self.assertEqual(hashes, [figure.file_sha256(p) for p in (self.phy_checkpoint, self.base_checkpoint)])
        for role in ('phy', 'baseline'):
            self.assertEqual(meta['checkpoint_' + role]['state_sha256_before'], meta['checkpoint_' + role]['state_sha256_after'])
        figure.render(out / 'responses_full_256.npz', out / 'mechanism_meta.json', out,
                      roi=[80, 80, 145, 145], dpi=60)
        self.assertTrue((out / 'panels_196/baseline_vs_hphy_roi.png').exists())
        self.assertFalse(json.loads((out / 'phy_stats.json').read_text())['available'])

    def test_checkpoint_after_real_dataset_construction_replays_without_protocol_mismatch(self):
        from mmseg.datasets.goji import GojiDataset
        saved = self.Config.fromstring(self.phy_cfg.pretty_text, '.py')
        target = dict(saved.data.train.target)
        target.pop('type')
        with patch('mmseg.datasets.custom.CustomDataset.__init__', return_value=None), \
                patch.object(GojiDataset, '_log_index_cfg'):
            GojiDataset(**target)
        self.assertIn('valid_pseudo_mask', saved.data.train.target.pipeline[-1]['keys'])
        self.assertNotEqual(self.phy_cfg.data, saved.data)
        checkpoint = torch.load(self.phy_checkpoint)
        checkpoint['meta']['config'] = saved.pretty_text
        torch.save(checkpoint, self.phy_checkpoint)
        original = self.phy_cfg.pretty_text
        out = self.replay('dataset_built_checkpoint')
        self.assertTrue((out / 'responses_full_256.npz').is_file())
        metadata = json.loads((out / 'phy_mechanism_meta.json').read_text())
        self.assertTrue(metadata['checkpoint_phy']['data_runtime_mask_normalized'])
        self.assertEqual(self.phy_cfg.pretty_text, original)
        self.phy_path = self.root / 'dataset_built_config.py'
        self.phy_path.write_text(saved.pretty_text, encoding='utf-8')
        out = self.replay('dataset_built_config')
        self.assertTrue((out / 'responses_full_256.npz').is_file())

    def test_real_data_protocol_changes_still_fail_before_building_model(self):
        saved = self.Config.fromstring(self.phy_cfg.pretty_text, '.py')
        saved.data.train.target.pipeline[-1]['keys'].append('valid_pseudo_mask')
        saved.data.val.split = 'different_validation_split.txt'
        checkpoint = torch.load(self.phy_checkpoint)
        checkpoint['meta']['config'] = saved.pretty_text
        torch.save(checkpoint, self.phy_checkpoint)
        with patch('mmseg.models.builder.build_train_model') as build:
            with self.assertRaisesRegex(ValueError, 'protocol differs from checkpoint in data'):
                self.api.load_model(self.phy_cfg, self.phy_checkpoint, True,
                                    torch.device('cpu'), self.root / 'must_not_build')
            build.assert_not_called()

    def test_early4k_and_historical_replay_restore_schedule_and_real_attenuation(self):
        from mmseg.models.uda.phy_reliability import hierarchical_phy_gate, phy_gate_parameters
        original = torch.load(self.phy_checkpoint)
        self.phy_checkpoint = self.root / 'hphy_iter_2000.pth'
        for mode, expected in (('early4k', .55), ('early3k', .7), ('historical', 0.)):
            config = self.Config.fromstring(self.phy_cfg.pretty_text, '.py')
            if mode == 'early3k':
                config.uda.phy_cfg.phy_gate_full_iter = 3000
            if mode == 'historical':
                for key in ('phy_gate_schedule', 'phy_gate_initial_strength', 'phy_gate_full_iter'):
                    config.uda.phy_cfg.pop(key)
                config.uda.phy_cfg.update(phy_warmup_iters=6000, phy_rampup_iters=4000)
            checkpoint = copy.deepcopy(original)
            checkpoint['meta'].update(iter=1999, config=config.pretty_text)
            torch.save(checkpoint, self.phy_checkpoint)
            out = self.replay(mode)
            meta = json.loads((out / 'phy_mechanism_meta.json').read_text())
            self.assertEqual(meta, json.loads((out / 'mechanism_meta.json').read_text()))
            self.assertAlmostEqual(meta['gate_rho_at_checkpoint'], expected)
            self.assertEqual(meta['checkpoint_phy']['historical_phy_schedule'], mode != 'early4k')
            with np.load(out / 'responses_full_256.npz') as arrays:
                response = {name: torch.from_numpy(arrays[key])[None, None]
                            for name, key in (('p_goji', 'P_G'), ('p_other_veg', 'P_V'))}
                direct = hierarchical_phy_gate(response, torch.from_numpy(arrays['teacher_pseudo_label'])[None],
                    torch.from_numpy(arrays['W_before'])[None], 2000,
                    valid_mask=torch.from_numpy(arrays['valid_mask'])[None],
                    return_details=True, **phy_gate_parameters(config.uda.phy_cfg))
                for key in ('S_phy', 'G_phy', 'A_phy', 'W_before', 'W_after'):
                    np.testing.assert_allclose(arrays[key], direct[key][0].numpy(), atol=1e-6)
                positive = arrays['positive_phy_weight_mask']
                np.testing.assert_allclose(arrays['A_phy'][positive],
                    1-arrays['W_after'][positive]/arrays['W_before'][positive], atol=1e-6)
                self.assertEqual(meta['gate_active_pixel_count'], int(arrays['gate_active_mask'].sum()))
                if mode == 'historical':
                    self.assertIsNone(meta['mean_A_phy_on_active_pixels'])
                    self.assertIsNotNone(meta['gate_empty_reason'])

    def test_replay_matches_live_probe_tensors_and_native_panel_pixels(self):
        from mmseg.utils.mechanism import load_probe_sample, probe_arrays
        checkpoint = torch.load(self.phy_checkpoint)
        running_config = self.Config.fromstring(self.phy_cfg.pretty_text, '.py')
        running_config.uda.pop('fixed_probe_filenames')
        running_config.uda.fixed_probe_max_samples = 1
        checkpoint['meta']['config'] = running_config.pretty_text
        torch.save(checkpoint, self.phy_checkpoint)
        out = self.replay()
        probe = self.api.sample_config(self.phy_cfg, self.target)
        with patch('mmseg.models.builder.build_train_model', side_effect=self.model_class):
            model, _ = self.api.load_model(self.phy_cfg, self.phy_checkpoint, True,
                                           torch.device('cpu'), out)
        sample, _ = load_probe_sample(probe, model)
        valid = torch.zeros(1, 1, 256, 256)
        valid[:, :, 30:226, 30:226] = 1
        live = probe_arrays(model, *sample, valid)
        with np.load(out / 'responses_full_256.npz') as saved:
            for offline, online in (('teacher_confidence', 'w_conf'), ('teacher_pseudo_label', 'pseudo_label'),
                    ('R_veg', 'r_veg'), ('Q_G', 'q_goji_given_veg'), ('S_phy', 's_phy_class_agreement'),
                    ('G_phy', 'g_phy'), ('A_phy', 'attenuation_phy'),
                    ('W_before', 'w_base'), ('W_after', 'w_t_phy'), ('teacher_top2_margin', 'teacher_top2_margin')):
                np.testing.assert_array_equal(saved[offline], live[online])
        figure.render(out / 'responses_full_256.npz', out / 'phy_mechanism_meta.json', out, dpi=60)
        figure.export_online_views(out / 'responses_full_256.npz', out / 'phy_mechanism_meta.json', out)
        with np.load(out / 'fixed_probe/selected_arrays/iter_012000.npz') as compatible:
            for key in set(compatible.files) & set(live):
                np.testing.assert_array_equal(compatible[key], live[key])
                self.assertEqual(compatible[key].dtype, live[key].dtype, key)
            self.assertEqual(json.loads(str(compatible['metadata']))['capture_mode'], 'checkpoint_replay')
        for size, folder in ((256, 'full256'), (196, 'center196')):
            for online, paper in (('target_image', 'a_target'), ('w_conf', 'teacher_confidence'),
                                 ('s_phy_class_agreement', 'c_s_phy'), ('attenuation_phy', 'd_attenuation'),
                                 ('pseudo_label', 'teacher_pseudo_label'), ('teacher_label', 'teacher_pseudo_label')):
                with Image.open(out / 'fixed_probe/snapshots' / folder / 'iter_012000' / (online+'.png')) as common, \
                        Image.open(out / ('panels_'+str(size)) / 'clean' / (paper+'.png')) as original:
                    np.testing.assert_array_equal(np.asarray(common.convert('RGB')),
                                                  np.asarray(original.convert('RGB')))

    def test_single_inference_entry_exports_responses_and_all_figures(self):
        output = self.root / 'combined'
        arguments = ['replay_hphy_mechanism.py', '--config-phy', str(self.phy_path),
            '--checkpoint-phy', str(self.phy_checkpoint), '--config-baseline', str(self.base_path),
            '--checkpoint-baseline', str(self.base_checkpoint), '--target-file', str(self.target),
            '--out-dir', str(output), '--device', 'cpu', '--dpi', '60',
            '--roi', '80', '80', '145', '145']
        with patch.object(sys, 'argv', arguments), \
                patch('mmseg.models.builder.build_train_model', side_effect=self.model_class):
            self.api.main()
        for name in ('responses_full_256.npz', 'responses_center_196.npz',
                     'phy_mechanism_meta.json', 'render_meta.json',
                     'phy_mechanism_1x5.png', 'phy_mechanism_1x5.pdf',
                     'panels_196/baseline_vs_hphy_roi.png', 'panels_256/clean/a_target.png',
                     'panels_196/teacher_pseudo_label.png',
                     'fixed_probe/selected_arrays/iter_012000.npz',
                     'fixed_probe/snapshots/iter_012000_phys.png',
                     'fixed_probe/snapshots/full256/iter_012000/attenuation_phy.png',
                     'fixed_probe/snapshots/center196/iter_012000/attenuation_phy.png',
                     'colorbars/attenuation.png', 'logs/replay.log', 'logs/render.log'):
            self.assertTrue((output / name).is_file(), name)
        self.assertEqual([(row[0], row[1]) for row in self.calls],
                         [('hphy', 'teacher'), ('hphy', 'student'), ('baseline', 'student')])
        with patch.object(sys, 'argv', arguments), \
                patch.object(self.api, 'replay', return_value=output), \
                patch.object(self.api, 'render', side_effect=RuntimeError('render failed')):
            with self.assertRaisesRegex(RuntimeError, 'render failed'):
                self.api.main()

    def test_short_checkpoint_command_uses_defaults_and_separate_output_directories(self):
        with patch.object(self.api, 'PHY_WORK_DIR', self.root / 'phy'), \
                patch.object(self.api, 'BASELINE_WORK_DIR', self.root / 'baseline'), \
                patch.object(self.api, 'OUTPUT_ROOT', self.root / 'analysis'), \
                patch.object(self.api, 'replay', return_value=self.root) as replay, \
                patch.object(self.api, 'render') as render, \
                patch.object(self.api, 'export_online_views') as common:
            self.api.main(['--checkpoint', 'iter_10000.pth'])
            first = replay.call_args_list[0].kwargs
            self.assertEqual(first['checkpoint_phy'], self.root / 'phy/iter_10000.pth')
            self.assertEqual(first['checkpoint_baseline'], self.root / 'baseline/latest.pth')
            self.assertEqual(first['config_phy'], self.api.CONFIG_PHY)
            self.assertEqual(first['config_baseline'], self.api.CONFIG_BASELINE)
            self.assertEqual(first['target_file'], self.api.TARGET_FILE)
            self.assertEqual(first['out_dir'].parent.parent, self.root / 'analysis')
            self.assertEqual([call.kwargs['target_file'].name for call in replay.call_args_list],
                ['3_1024_1280_1280_1536.tif', '3_256_1152_512_1408.tif',
                 '3_1536_512_1792_768.tif', '3_1536_1024_1792_1280.tif',
                 '3_1792_1280_2048_1536.tif', '3_1536_640_1792_896.tif'])
            self.assertEqual(len({call.kwargs['out_dir'] for call in replay.call_args_list}), 6)
            self.api.main(['--checkpoint-phy', 'iter_10000.pth',
                           '--checkpoint-baseline', 'iter_15000.pth'])
            second = replay.call_args.kwargs
            self.assertEqual(second['checkpoint_baseline'], self.root / 'baseline/iter_15000.pth')
            self.assertNotEqual(first['out_dir'].parent, second['out_dir'].parent)
            self.assertEqual(render.call_count, 12)
            self.assertEqual(common.call_count, 12)

    def test_multiple_images_keep_responses_weights_and_figures_independent(self):
        second = self.target.with_name('3_256_1152_512_1408.tif')
        second_rgb = np.flip(fixture_arrays()[0]['image_rgb'], axis=1).copy()
        Image.fromarray(second_rgb).save(second)
        for name in ('ndvi', 'ndre', 'cire'):
            Image.fromarray(np.ones((256, 256), np.float32)).save(
                self.root / ('30/' + name + '_dir/val') / second.name)
        network_class = type(self.model_class(self.phy_cfg).get_ema_model())
        original = network_class.generate_pseudo_label

        def generate(network, image, metas):
            logits = original(network, image, metas)
            return torch.zeros_like(logits) if Path(metas[0]['filename']).name == second.name else logits

        output = self.root / 'multi'
        arguments = ['--config-phy', str(self.phy_path), '--checkpoint', str(self.phy_checkpoint),
            '--config-baseline', str(self.base_path), '--checkpoint-baseline', str(self.base_checkpoint),
            '--target-file', str(self.target), str(second), '--out-dir', str(output),
            '--device', 'cpu', '--dpi', '60']
        with patch('mmseg.models.builder.build_train_model', side_effect=self.model_class), \
                patch.object(network_class, 'generate_pseudo_label', generate):
            self.api.main(arguments)
        for target in (self.target, second):
            folder = output / target.stem
            for name in ('phy_mechanism_1x5.png', 'phy_mechanism_1x5.pdf',
                         'panels_196/clean/c_s_phy.png', 'panels_256/clean/d_attenuation.png',
                         'phy_stats.json', 'phy_mechanism_meta.json'):
                self.assertTrue((folder / name).is_file(), name)
            metadata = json.loads((folder / 'phy_mechanism_meta.json').read_text())
            self.assertEqual(metadata['sample_id'], target.name)
            with np.load(folder / 'responses_full_256.npz') as full, \
                    np.load(folder / 'responses_center_196.npz') as center:
                with Image.open(target) as source:
                    np.testing.assert_array_equal(full['image_rgb'], np.asarray(source))
                np.testing.assert_array_equal(center['A_phy'], full['A_phy'][30:226, 30:226])
                expected_weight = 0. if target == second else 1.
                np.testing.assert_allclose(full['W_before'][30:226, 30:226], expected_weight)
                np.testing.assert_allclose(full['W_after'], full['W_before'] * full['G_phy'])
        self.assertEqual(len(self.calls), 6)
        self.assertTrue(all(call[5] == (1, 6, 256, 256) for call in self.calls))

    def test_multiple_images_reject_duplicate_names_and_single_reference_gt(self):
        with patch.object(self.api, 'replay') as replay:
            with self.assertRaises(SystemExit):
                self.api.main(['--target-file', str(self.target), str(self.target)])
            with self.assertRaises(SystemExit):
                self.api.main(['--gt-file', 'one_reference.tif'])
            replay.assert_not_called()

    def test_iteration_fallback_and_conflicting_sources(self):
        resolve = self.api.resolve_iteration
        arbitrary = self.root / 'renamed.pth'
        arbitrary.write_bytes(self.phy_checkpoint.read_bytes())
        result = resolve(arbitrary, {'iter': 11999})
        self.assertEqual(result['completed_iter'], 12000)
        self.assertIn('byte-identical', result['iteration_source'])
        arbitrary.write_bytes(b'fixture without a numbered alias')
        result = resolve(arbitrary, {'iter': 1999}, explicit_iter=2000)
        self.assertIn('explicit --iter', result['iteration_source'])
        record = {'mIoU': dict(iter=2000, path=str(arbitrary))}
        (self.root / 'checkpoint_best.json').write_text(json.dumps(record))
        self.assertEqual(resolve(arbitrary, {'iter': 1999})['completed_iter'], 2000)
        for path, meta, explicit in ((arbitrary, {'iter': 1999}, 3000),
                (self.phy_checkpoint, {'iter': 11999}, 3000),
                (self.phy_checkpoint, {'iter': 3000}, None)):
            with self.assertRaisesRegex(ValueError, 'disagree'):
                resolve(path, meta, explicit)

    def test_training_rejects_historical_schedule_but_accepts_matching_checkpoint(self):
        from tools.doris_checkpoint_plan import check_hphy_training_checkpoint
        config = copy.deepcopy(self.phy_cfg)
        config.resume_from = str(self.phy_checkpoint)
        check_hphy_training_checkpoint(config)
        checkpoint = torch.load(self.phy_checkpoint)
        saved = self.Config.fromstring(self.phy_cfg.pretty_text, '.py')
        saved.uda.phy_cfg.phy_gate_full_iter = 3000
        checkpoint['meta']['config'] = saved.pretty_text
        torch.save(checkpoint, self.phy_checkpoint)
        with self.assertRaisesRegex(ValueError, 'schedule mismatch'):
            check_hphy_training_checkpoint(config)
        config.resume_from = None
        config.load_from = str(self.phy_checkpoint)
        with self.assertRaisesRegex(ValueError, 'schedule mismatch'):
            check_hphy_training_checkpoint(config)

    def test_gt_is_read_only_after_both_forwards_and_reduced(self):
        from mmseg.datasets.pipelines import LoadAnnotations
        gt_file = self.root / 'gt.tif'
        raw = np.ones((256, 256), np.uint8)
        raw[:50] = 0
        raw[50:100] = 2
        Image.fromarray(raw).save(gt_file)
        real_loader = LoadAnnotations
        calls = self.calls
        class TrackingLoader(real_loader):
            def __call__(self, result):
                calls.append(('gt',))
                return super().__call__(result)
        with patch('mmseg.datasets.pipelines.LoadAnnotations', TrackingLoader):
            output = self.replay(gt_file=gt_file)
        self.assertEqual(self.calls[-1], ('gt',))
        with np.load(output / 'responses_full_256.npz') as arrays:
            self.assertTrue((arrays['gt_label'][:50] == 255).all())
            self.assertTrue((arrays['gt_label'][50:100] == 1).all())

    def test_missing_head_legacy_iteration_and_vi_fail_explicitly(self):
        original = torch.load(self.phy_checkpoint)
        bad = copy.deepcopy(original)
        bad['state_dict'].pop('phy_head.gv_head.weight')
        torch.save(bad, self.phy_checkpoint)
        with self.assertRaisesRegex(ValueError, 'missing='):
            self.replay('missing_head')
        bad = copy.deepcopy(original)
        bad['state_dict'] = {key: value for key, value in bad['state_dict'].items() if not key.startswith('phy_head.')}
        bad['state_dict']['phy_head.layers.0.weight'] = torch.zeros(1)
        torch.save(bad, self.phy_checkpoint)
        with self.assertRaisesRegex(RuntimeError, 'mode mismatch'):
            self.replay('legacy')
        with self.assertRaisesRegex(ValueError, 'unambiguous'):
            self.api.resolve_iteration(self.root / 'ambiguous.pth', {'iter': 6000})
        torch.save(original, self.phy_checkpoint)
        (self.root / '30/ndre_dir/val/SYNTHETIC.tif').unlink()
        with self.assertRaises(FileNotFoundError):
            self.replay('missing_vi')

    def test_old_protocol_wrong_modules_and_wrong_input_channels_fail(self):
        from tools.check_doris_stage1_ablation_configs import audit
        self.assertEqual(len(audit(lambda path: self.Config.fromfile(path))), 24)
        config = copy.deepcopy(self.phy_cfg)
        config.runner.max_iters = 20000
        with self.assertRaisesRegex(ValueError, '15K'):
            self.api.verify_protocol(config, True)
        config = copy.deepcopy(self.phy_cfg)
        config.uda.struct_cfg.enable = True
        with self.assertRaisesRegex(ValueError, 'phy-only'):
            self.api.verify_protocol(config, True)
        with self.assertRaisesRegex(ValueError, 'aligned full'):
            self.api.infer_model(self.model_class(self.phy_cfg),
                (torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), [{}]), {}, True)


if __name__ == '__main__':
    unittest.main()
