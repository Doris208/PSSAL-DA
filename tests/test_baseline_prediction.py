"""Baseline exports and comparison alignment using explicitly synthetic fixtures."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_minimal_output import static_config
from tools import replay_baseline_prediction as baseline
from tools import build_branch_prediction_comparison as comparison


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


def fixture():
    y, x = np.mgrid[:256, :256]
    rgb = np.stack([x, y, (x+y)//2], -1).astype(np.uint8)
    prediction = np.minimum(x//52, 4).astype(np.uint8)
    image = np.concatenate([rgb.astype(np.float32), np.zeros((256, 256, 3), np.float32)], -1)
    valid = np.zeros((256, 256), bool)
    valid[30:226, 30:226] = True
    arrays = dict(image_rgb=rgb, baseline_prediction=prediction, image_6ch=image, valid_mask=valid)
    checkpoint = dict(path='/synthetic/iter_4000.pth', sha256='a'*64, completed_iter=4000,
        state_sha256_before='b'*64, state_sha256_after='b'*64, eval=True, inference_mode=True)
    inference = dict(model_identity='Student', test_cfg=dict(mode='slide'), scales=[1, .5],
        hr_crop_size=[128, 128], hr_slide_inference=True, rescale=True, fp16_enabled=False, flip=False)
    meta = dict(sample_id='synthetic.tif', target_file='/synthetic/rgb_dir/val/synthetic.tif',
        target_sha256='c'*64, direction='goji26RGB2goji30RGB',
        vi_files={name: dict(path='/synthetic/'+name, sha256='d'*64) for name in ('ndvi', 'ndre', 'cire')},
        class_palette=baseline.PALETTE.tolist(), class_names=list(baseline.GOJI['GOJI_CLASSES']),
        original_shape=[256, 256], center_shape=[196, 196], crop=list(baseline.CROP),
        reduce_zero_label=True, ignore_index=255, checkpoint_baseline=checkpoint,
        input_channel_order=['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE'],
        target_normalization=dict(mean=[0]*6, std=[1]*6), vi_normalization={},
        student_inference=inference, prediction_network='Student', synthetic_fixture=True)
    return arrays, meta


def paired_artifact(root, role, arrays, meta):
    out = root/role
    out.mkdir()
    arrays, meta = copy.deepcopy(arrays), copy.deepcopy(meta)
    field = 'hphy_prediction' if role == 'phy' else 'struct_prediction'
    arrays[field] = arrays.pop('baseline_prediction')
    meta['checkpoint_'+role] = meta.pop('checkpoint_baseline')
    meta['checkpoint_'+role]['completed_iter'] = 12000
    path = out/'responses_full_256.npz'
    np.savez_compressed(path, **arrays)
    meta['arrays_sha256'] = {path.name: baseline.file_sha256(path)}
    filename = 'phy_mechanism_meta.json' if role == 'phy' else 'struct_mechanism_meta.json'
    (out/filename).write_text(json.dumps(meta), encoding='utf-8')
    return path


class BaselinePredictionTests(unittest.TestCase):
    def test_baseline_config_preprocessing_and_cli(self):
        cfg = static_config(baseline.CONFIG.relative_to(ROOT).as_posix())
        self.assertFalse(any(cfg['uda'][key]['enable'] for key in ('phy_cfg', 'struct_cfg', 'proto_cfg', 'struct_edge_cfg')))
        for suffix in ('_phy_gate', '_struct_gate'):
            other = static_config(baseline.CONFIG.relative_to(ROOT).as_posix().replace('_3vi.py', '_3vi'+suffix+'.py'))
            for key in ('target_img_norm_cfg', 'data'):
                self.assertEqual(cfg[key], other[key])
            for key in ('type', 'scales', 'hr_crop_size', 'hr_slide_inference', 'test_cfg'):
                self.assertEqual(cfg['model'][key], other['model'][key])
        self.assertEqual(baseline.CHECKPOINT.name, 'iter_4000.pth')
        for script in ('replay_baseline_prediction.py', 'build_branch_prediction_comparison.py'):
            result = subprocess.run([sys.executable, str(ROOT/'tools'/script), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        with patch.object(baseline, 'replay', return_value=Path('/synthetic/output')) as run:
            baseline.main(['--config', 'a.py', '--checkpoint', 'b.pth', '--out-dir', 'result',
                           '--device', 'cpu', '--target-file', 'one.tif', 'two.tif'])
            self.assertEqual(run.call_args.kwargs['target_files'], [Path('one.tif'), Path('two.tif')])

    def test_native_ids_crop_palette_pdf_and_no_overwrite(self):
        arrays, meta = fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            out = baseline.export_sample(arrays, meta, Path(folder)/'output', dpi=40)
            np.testing.assert_array_equal(np.load(out/'baseline_prediction_full_256.npy'), arrays['baseline_prediction'])
            for name, expected in (
                    ('baseline_prediction_full_256.png', arrays['baseline_prediction']),
                    ('baseline_prediction_color_256.png', baseline.PALETTE[arrays['baseline_prediction']]),
                    ('baseline_prediction_color_196.png', baseline.PALETTE[arrays['baseline_prediction'][30:226, 30:226]]),
                    ('target_rgb_196.png', arrays['image_rgb'][30:226, 30:226])):
                with Image.open(out/name) as image:
                    np.testing.assert_array_equal(np.asarray(image), expected)
            self.assertTrue((out/'baseline_prediction_panel.pdf').read_bytes().startswith(b'%PDF'))
            metadata = json.loads((out/'baseline_meta.json').read_text())
            self.assertEqual(metadata['arrays_sha256']['baseline_prediction.npz'], baseline.file_sha256(out/'baseline_prediction.npz'))
            with self.assertRaises(FileExistsError):
                baseline.export_sample(arrays, meta, out, dpi=40)
            for invalid in (np.full((256, 256), 255, np.uint8), np.zeros((196, 196), np.uint8),
                            np.zeros((256, 256), np.float32)):
                with self.assertRaises(ValueError):
                    baseline.export_sample(dict(arrays, baseline_prediction=invalid), meta, Path(folder)/'bad')

    def test_single_and_six_samples_use_one_baseline_load(self):
        cfg = attributes(static_config(baseline.CONFIG.relative_to(ROOT).as_posix()))
        cfg.pretty_text = 'SYNTHETIC CONFIG'
        arrays, _ = fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary:
            root = Path(temporary)
            targets = []
            for index in range(6):
                target = root/'30/rgb_dir/val'/('synthetic_'+str(index)+'.tif')
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arrays['image_rgb']).save(target)
                targets.append(target)
                for name in ('ndvi', 'ndre', 'cire'):
                    vi = root/'30'/(name+'_dir')/'val'/target.name
                    vi.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.zeros((256, 256), np.float32)).save(vi)
            checkpoint = root/'iter_4000.pth'
            checkpoint.write_bytes(b'SYNTHETIC mocked checkpoint')
            info = dict(completed_iter=4000, sha256=baseline.file_sha256(checkpoint), synthetic_fixture=True)
            sample_image = Mock()
            sample_image.__getitem__ = Mock(return_value=SimpleNamespace(numpy=lambda: arrays['image_6ch'].transpose(2, 0, 1)))
            sample_vi = Mock()
            sample_vi.__getitem__ = Mock(return_value=SimpleNamespace(numpy=lambda: np.zeros((256, 256), np.float32)))
            sample = (sample_image, sample_vi, [dict(img_norm_cfg=dict(mean=[0]*6, std=[1]*6))])
            from types import ModuleType
            mechanism = ModuleType('mmseg.utils.mechanism')
            mechanism.load_probe_sample = Mock(return_value=(sample, {}))
            mechanism.student_test_metadata = Mock(return_value=fixture()[1]['student_inference'])
            shared = ModuleType('tools.replay_hphy_mechanism')
            shared.TARGET_FILES = targets
            shared.verify_protocol = Mock()
            shared.sample_config = Mock(return_value=dict(margins=[30]*4))
            shared.load_model = Mock(return_value=(object(), info))
            shared.infer_model = Mock(return_value=(dict(baseline_prediction=arrays['baseline_prediction']), 'state'))
            modules = {'torch': SimpleNamespace(device=lambda _: SimpleNamespace(type='cpu')),
                'mmcv': SimpleNamespace(Config=SimpleNamespace(fromfile=lambda _: copy.deepcopy(cfg))),
                'mmseg.utils.mechanism': mechanism, 'tools.replay_hphy_mechanism': shared}
            import tools
            with patch.dict(sys.modules, modules), patch.object(tools, 'replay_hphy_mechanism', shared, create=True):
                for count in (1, 6):
                    out = baseline.replay(baseline.CONFIG, checkpoint, root/('run'+str(count)), 'cpu',
                                          targets[:1] if count == 1 else None, dpi=30)
                    summary = json.loads((out/'baseline_run_summary.json').read_text())
                    self.assertEqual(summary['sample_count'], count)
                    self.assertEqual([row['sample_id'] for row in summary['samples']], [path.name for path in targets[:count]])
                    self.assertTrue((out/'logs/replay.log').is_file())
                    for target in targets[:count]:
                        meta = json.loads((out/target.stem/'baseline_meta.json').read_text())
                        self.assertEqual(meta['dataset_split'], 'val')
                        self.assertEqual(meta['prediction_network'], 'Student')
                        self.assertFalse(meta['gt_loaded'])
                self.assertEqual(shared.load_model.call_count, 2)
                self.assertEqual(shared.infer_model.call_count, 7)
                with self.assertRaises(FileExistsError):
                    baseline.replay(baseline.CONFIG, checkpoint, root/'run6', 'cpu', targets)
                shared.load_model.return_value = (object(), dict(info, completed_iter=12000))
                with self.assertRaisesRegex(ValueError, 'Iter 4000'):
                    baseline.replay(baseline.CONFIG, checkpoint, root/'wrong_iter', 'cpu', targets[:1])


class ComparisonTests(unittest.TestCase):
    def test_comparison_alignment_and_optional_gt(self):
        arrays, meta = fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            root = Path(folder)
            base = baseline.export_sample(arrays, meta, root/'baseline', dpi=30)/'baseline_prediction.npz'
            phy = paired_artifact(root, 'phy', arrays, meta)
            structural = paired_artifact(root, 'struct', arrays, meta)
            out = comparison.render(base, phy, structural, root/'no_gt', dpi=30)
            self.assertFalse((out/'tile_metrics.json').exists())
            with self.assertRaises(FileExistsError):
                comparison.render(base, phy, structural, out, dpi=30)
            gt = root/'synthetic.png'
            Image.fromarray(arrays['baseline_prediction']+1).save(gt)
            comparison.render(base, phy, structural, root/'with_gt', gt_file=gt, dpi=30)
            stats = json.loads((root/'with_gt/tile_metrics.json').read_text())
            self.assertEqual(stats['scopes']['full256']['models']['baseline']['mIoU'], 1.)
            artifacts = [comparison.load_artifact(path, role) for path, role in zip((base, phy, structural), comparison.ROLES)]
            changed = copy.deepcopy(artifacts)
            changed[2][1]['student_inference']['test_cfg'] = dict(mode='whole')
            with self.assertRaisesRegex(ValueError, 'inference setting'):
                comparison.verify_alignment(changed)
            changed = copy.deepcopy(artifacts)
            changed[1][1].pop('student_inference')
            comparison.verify_alignment(changed)
            for key, value in (('image_6ch', 1), ('image_rgb', 0), ('valid_mask', False)):
                changed = copy.deepcopy(artifacts)
                changed[1][0][key][80, 80] = value
                with self.assertRaisesRegex(ValueError, 'pixels or spatial'):
                    comparison.verify_alignment(changed)
            for key, value in (('target_sha256', 'wrong'), ('crop', [0, 0, 196, 196]),
                               ('prediction_network', 'EMA Teacher'), ('class_palette', [])):
                path = phy.parent/'phy_mechanism_meta.json'
                original = json.loads(path.read_text())
                path.write_text(json.dumps(dict(original, **{key: value})))
                with self.assertRaises(ValueError):
                    comparison.render(base, phy, structural, root/'reject', dpi=30)
                self.assertFalse((root/'reject').exists())
                path.write_text(json.dumps(original))

    def test_layout_iterations_and_absent_classes(self):
        _, meta = fixture()
        pixels = np.zeros((196, 196, 3), np.uint8)
        for count in (4, 5):
            panels = [('Target RGB', pixels)]+[
                ('Baseline\nStudent\nIter 4000', pixels) for _ in range(count-1)]
            fig = comparison.comparison_figure(panels, synthetic=True)
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                for axis in fig.axes:
                    self.assertEqual(axis.images[0].get_interpolation(), 'nearest')
                    bounds = axis.title.get_window_extent(renderer)
                    self.assertLessEqual(bounds.y1, fig.bbox.y1)
                    self.assertGreaterEqual(bounds.x0, 0)
                    self.assertLessEqual(bounds.x1, fig.bbox.x1)
            finally:
                baseline.plt.close(fig)
        truth = np.zeros((256, 256), np.uint8)
        stats = comparison.gt_statistics(truth, [truth]*3)
        values = stats['scopes']['full256']['models']['baseline']
        self.assertEqual(values['mIoU'], 1.)
        self.assertIsNone(values['iou']['road'])
        truth[:] = 255
        self.assertIsNone(comparison.gt_statistics(truth, [truth]*3)['scopes']['center196']['models']['phy']['mIoU'])
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            path = Path(folder)/'synthetic.png'
            raw = np.tile(np.array([0, 1, 2, 3, 4, 5, 255, 0], np.uint8), (256, 32))
            Image.fromarray(raw).save(path)
            reduced = comparison.load_raw_gt(path, 'synthetic.tif')
            np.testing.assert_array_equal(reduced[0, :8], [255, 0, 1, 2, 3, 4, 255, 255])


if __name__ == '__main__':
    unittest.main()
