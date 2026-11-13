"""IRRG26 upgrade checks, with explicit synthetic inputs and historical regressions."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from test_minimal_output import ROOT, static_config, load_methods
from test_baseline_prediction import attributes
from test_struct_probe_cpu import load_function
from tools.check_doris_stage1_ablation_configs import audit, MODULES, PROTOCOL
from tools.check_doris_stage1_configs import VARIANTS
from tools.doris_probe_data import replay_direction

DIRECTION = 'goji26IRRG2goji30RGB'
REFERENCE = 'goji26RGB2goji30RGB'
COMMIT = '1377b878049d8d37edafdf4ed339ed1540e2169f'


def config(suffix='', direction=DIRECTION):
    return attributes(static_config('configs/hrda/doris/' + direction + '/doris_' + direction +
                                    '_hrda_3vi' + suffix + '.py'))


class ConfigUpgradeTests(unittest.TestCase):
    def test_24_configs_and_both_eight_command_launchers(self):
        self.assertEqual(len(audit(lambda path: attributes(static_config(path)))), 24)

    def test_parameters_modules_protocol_and_direction_only_differences(self):
        for suffix, flags in VARIANTS:
            cfg, rgb = config(suffix), config(suffix, REFERENCE)
            with self.subTest(suffix=suffix):
                self.assertEqual(replay_direction(cfg), DIRECTION)
                self.assertEqual(tuple(cfg.uda[module + '_cfg'].enable for module in MODULES), flags)
                for module, enabled in zip(MODULES, flags):
                    if enabled:
                        current = dict(cfg.uda[module + '_cfg'])
                        reference = dict(rgb.uda[module + '_cfg'])
                        if module == 'struct_edge':
                            self.assertEqual(current.pop('boundary_pair_min_weight'), .80)
                            reference.pop('boundary_pair_min_weight', None)
                        self.assertEqual(current, reference, module)
                self.assertEqual(cfg.model, rgb.model)
                for key in PROTOCOL:
                    if key not in ('data', 'source_img_norm_cfg'):
                        self.assertEqual(cfg[key], rgb[key], key)
                self.assertEqual(cfg.data.train.target, rgb.data.train.target)
                self.assertEqual(cfg.data.val, rgb.data.val)
                self.assertEqual(cfg.data.test, rgb.data.test)
                self.assertEqual(cfg.data.train.source.img_dir, 'img_dir/train')
                self.assertEqual(cfg.source_img_norm_cfg.mean[:3], [111.366, 122.724, 135.261])
                self.assertEqual(cfg.source_img_norm_cfg.std[:3], [61.302, 62.369, 55.989])
                self.assertEqual(cfg.source_img_norm_cfg.channel_order, [2, 1, 0, 3, 4, 5])
                self.assertNotEqual(cfg.source_img_norm_cfg, rgb.source_img_norm_cfg)
                self.assertEqual(cfg.runner.max_iters, 15000)
                self.assertEqual((cfg.seed, cfg.data.samples_per_gpu, cfg.optimizer.lr), (0, 2, 6e-5))
                self.assertEqual((cfg.uda.alpha, cfg.uda.mix, cfg.uda.pseudo_threshold), (.999, 'class', .968))
                self.assertIsNone(cfg.get('load_from'))
                self.assertIsNone(cfg.get('resume_from'))
                self.assertEqual(cfg.uda.source_input_channel_order, ['NIR', 'R', 'G', 'NDVI', 'NDRE', 'CIRE'])
                if any(flags):
                    self.assertEqual(cfg.uda.fixed_probe_max_samples, 6 if flags[0] else 1)
                    if flags[0]:
                        self.assertEqual(cfg.uda.fixed_probe_filenames[0], '3_1536_640_1792_896.tif')
                        self.assertEqual(set(cfg.uda.fixed_probe_filenames), set(rgb.uda.fixed_probe_filenames))

    def test_other_directions_common_data_and_algorithms_match_reference_commit(self):
        paths = list((ROOT / 'configs/hrda/doris' / REFERENCE).glob('*.py'))
        paths += list((ROOT / 'configs/hrda/doris/goji30IRRG2goji26IRRG').glob('*.py'))
        paths += [ROOT / name for name in ('configs/hrda/pssalda_26irrg2_30rgb_ablation_common.py',
            'configs/hrda/pssalda_26rgb2_30rgb_ablation_common.py', 'mmseg/models/uda/struct_prior_gate.py',
            'mmseg/models/uda/phy_reliability.py', 'mmseg/models/uda/proto_gate.py',
            'mmseg/models/uda/dacs.py', 'mmseg/models/losses/struct_edge_loss.py')]
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            before = subprocess.check_output(['git', 'show', COMMIT + ':' + relative], cwd=ROOT).decode('utf-8')
            self.assertEqual(path.read_text(encoding='utf-8'), before.replace('\r\n', '\n'), relative)
        script = 'train_doris_mitb5_stage1_ablations_20k.sh'
        before = subprocess.check_output(['git', 'show', COMMIT + ':' + script], cwd=ROOT).decode('utf-8')
        current = (ROOT / script).read_text(encoding='utf-8')
        unaffected = lambda text: [line for line in text.splitlines() if DIRECTION not in line]
        self.assertEqual(unaffected(before), unaffected(current))

    def test_protocol_rejects_wrong_data_direction_and_preserves_historical_v1(self):
        verify = load_function(ROOT / 'tools/replay_hphy_mechanism.py', 'verify_protocol', dict(Path=Path))
        for direction in (DIRECTION, REFERENCE):
            for suffix, phy, struct in (('', False, False), ('_phy_gate', True, False), ('_struct_gate', False, True)):
                cfg = config(suffix, direction)
                self.assertEqual(verify(cfg, phy, struct), direction)
                if struct:
                    cfg.uda.struct_cfg.fusion_mode = 'doris_confidence_guided'
                    self.assertEqual(verify(cfg, phy, struct), direction)
                cfg.data.train.target.img_dir = 'img_dir/train'
                with self.assertRaises(ValueError):
                    verify(cfg, phy, struct)
        cfg = config()
        cfg.name = cfg.name.replace(DIRECTION, REFERENCE)
        with self.assertRaises(ValueError):
            verify(cfg, False)

    def test_paired_cli_derives_irrg_configs_paths_and_first_sample(self):
        function = load_function(ROOT / 'tools/replay_hphy_mechanism.py', 'resolve_cli_inputs',
            dict(ROOT=ROOT, Path=Path, RGB30_DIRECTIONS=(REFERENCE, DIRECTION), replay_direction=replay_direction,
                 TARGET_FILES=[Path(name) for name in config('_phy_gate', REFERENCE).uda.fixed_probe_filenames],
                 BASELINE_WORK_DIR=Path('/historical/rgb_baseline')))
        parser = SimpleNamespace(fromfile=lambda path: config(
            '_phy_gate' if '_phy_gate' in str(path) else '', DIRECTION))
        args = dict(direction=DIRECTION, config_phy=None, config_baseline=None,
            checkpoint_phy=Path('/checkpoints/phy.pth'), checkpoint_baseline=Path('/checkpoints/base.pth'),
            target_file=None, out_dir=Path('/new/output'))
        with patch.dict(sys.modules, {'mmcv': SimpleNamespace(Config=parser)}):
            function(args, 'phy', Path('/historical/rgb_phy'), Path('/historical/figures'))
        self.assertIn(DIRECTION, str(args['config_baseline']))
        self.assertEqual(args['checkpoint_baseline'], Path('/checkpoints/base.pth'))
        self.assertEqual(args['target_file'][0].name, '3_1536_640_1792_896.tif')

    def test_all_v2_six_channel_normalizers_preserve_stored_band_and_vi_order(self):
        import os.path as osp
        from io import BytesIO
        vi_helpers = runpy.run_path(str(ROOT / 'mmseg/vi_utils.py'))
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            folder = Path(temporary)
            pixels = np.full((256, 256, 3), [23, 87, 149], dtype=np.uint8)
            tif = folder / 'synthetic.tif'
            Image.fromarray(pixels).save(tif)
            class FileClient:
                def __init__(self, **kwargs): pass
                def get(self, path): return Path(path).read_bytes()
            mmcv = SimpleNamespace(FileClient=FileClient, print_log=lambda *a, **kw: None,
                # Match MMCV's default OpenCV BGR decode in the lightweight harness.
                imfrombytes=lambda data, **kw: np.asarray(Image.open(BytesIO(data)))[..., ::-1].copy(),
                imnormalize=lambda data, mean, std, to_rgb: (data.astype(np.float32) - mean) / std)
            loader = load_methods('mmseg/datasets/pipelines/loading.py', 'LoadImageFromFile',
                ['__init__', '__call__'], dict(mmcv=mmcv, np=np, osp=osp))
            indices = load_methods('mmseg/datasets/pipelines/loading.py', 'LoadGojiIndices',
                ['__init__', '__call__', '_load_index', '_normalize_index'],
                dict(mmcv=mmcv, np=np, osp=osp, Image=Image, BytesIO=BytesIO))
            normalizer = load_methods('mmseg/datasets/pipelines/transforms.py', 'Normalize',
                ['__init__', '__call__'], dict(mmcv=mmcv, np=np))
            prefixes = {}
            for name, value in (('ndvi', .2), ('ndre', .4), ('cire', 2.)):
                directory = folder / name
                directory.mkdir()
                Image.fromarray(np.full((256, 256), value, np.float32)).save(directory / tif.name)
                prefixes[name] = str(directory)
            for norm in (config().source_img_norm_cfg, config().target_img_norm_cfg):
                results = dict(img_prefix=str(folder), img_info=dict(filename=tif.name), index_prefixes=prefixes,
                    use_indices=True, use_vi=True, vi_as_input=True, vi_norm_for_input=False,
                    vi_names=['ndvi', 'ndre', 'cire'], vi_norm_cfg=config().vi_norm_cfg)
                modules = {'mmseg.vi_utils': SimpleNamespace(**vi_helpers)}
                with patch.dict(sys.modules, modules):
                    results = normalizer(**norm)(indices()(loader()(results)))
                restored = results['img'] * np.asarray(norm['std']) + np.asarray(norm['mean'])
                np.testing.assert_allclose(restored[..., :3], pixels, atol=1e-3)
                np.testing.assert_allclose(restored[0, 0, 3:], [.2, .4, 2.], atol=1e-6)
                np.testing.assert_allclose(results['vi_img'][0, 0], [.6, .7, .4], atol=1e-6)


class IRRGFigureTests(unittest.TestCase):
    def test_irrg_hphy_and_v2_struct_figures_keep_raw_scale_and_masks(self):
        from test_hphy_mechanism import fixture_arrays as phy_fixture
        from test_struct_mechanism import fixture_arrays as struct_fixture
        from tools import build_hphy_mechanism_figure as phy
        from tools import build_struct_mechanism_figure as structural
        from tools.doris_struct_evidence import prepare_struct_evidence
        arrays, meta = phy_fixture()
        meta['direction'] = DIRECTION
        phy.validate_arrays(arrays, meta)
        arrays, meta = struct_fixture()
        meta['direction'] = DIRECTION
        meta['gate'].update(fusion_mode='struct_distinct_v1', confidence_weight=.15,
            continuity_weight=.30, adjacency_weight=.30, semantic_weight=.25,
            struct_gate_schedule='early_nonzero_linear', struct_gate_initial_strength=.10,
            struct_gate_full_iter=4000)
        valid = arrays['teacher_pseudo_label'] < 5
        arrays.update(confidence_available=valid, continuity_available=valid,
            evidence_count=2 + arrays['adjacency_available'].astype(int) + arrays['semantic_available'].astype(int),
            target_cross_neighbor_count=np.ones((256, 256), np.float32), source_cross_prior_available=valid)
        a, s = arrays['adjacency_available'], arrays['semantic_available']
        arrays['R_struct'] = (.15 * arrays['R_conf'] + .30 * arrays['R_loc'] +
            .30 * arrays['R_adj'] * a + .25 * arrays['R_prob'] * s) / (.45 + .30 * a + .25 * s + 1e-6)
        gate = np.maximum(arrays['R_struct'], .65)
        gate = np.where(arrays['R_struct'] < .32, gate * .5, gate)
        arrays['G_struct'] = np.where(np.isin(arrays['teacher_pseudo_label'], [2, 3, 4]), gate, 1).astype(np.float32)
        arrays['W_after'] = arrays['W_before'] * arrays['G_struct']
        arrays['delta_W'] = arrays['W_after'] - arrays['W_before']
        arrays['A_struct'] = 1 - arrays['G_struct']
        arrays['below_threshold_mask'] = arrays['teacher_struct_mask'] & (arrays['R_struct'] < .32)
        structural.validate_arrays(arrays, meta)
        _, checked = prepare_struct_evidence(arrays, meta)
        self.assertTrue(checked['struct_validation']['passed'])
        with self.assertRaises(ValueError):
            structural.validate_arrays(arrays, dict(meta, gate=dict(meta['gate'], fusion_mode='doris_confidence_guided')))

    def test_irrg_comparison_records_actual_different_iterations(self):
        from test_baseline_prediction import fixture, paired_artifact
        from tools import replay_baseline_prediction as baseline
        from tools import build_branch_prediction_comparison as compare
        arrays, meta = fixture()
        meta['direction'] = DIRECTION
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            root = Path(temporary)
            base = baseline.export_sample(arrays, meta, root / 'baseline', dpi=30) / 'baseline_prediction.npz'
            phy = paired_artifact(root, 'phy', arrays, meta)
            struct = paired_artifact(root, 'struct', arrays, meta)
            path = phy.parent / 'phy_mechanism_meta.json'
            saved = json.loads(path.read_text())
            saved['checkpoint_phy']['completed_iter'] = 15000
            path.write_text(json.dumps(saved))
            out = compare.render(base, phy, struct, root / 'comparison', dpi=30)
            metadata = json.loads((out / 'comparison_meta.json').read_text())
            self.assertEqual(metadata['direction'], DIRECTION)
            self.assertEqual(metadata['checkpoint_iterations']['phy'], 15000)
            self.assertFalse(metadata['matched_training_iterations'])


if __name__ == '__main__':
    unittest.main()
