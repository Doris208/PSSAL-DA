"""Synthetic online six-image probes and exact native-grid paper exports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from test_minimal_output import ROOT, static_config, torch
from test_hphy_mechanism import fixture_arrays
from tools import build_doris_mechanism_figures as plot


def online_maps():
    arrays, _ = fixture_arrays()
    maps = dict(target_image=arrays['image_rgb'].astype(np.float32)/255,
        valid_mask=arrays['valid_mask'], w_conf=arrays['teacher_confidence'],
        pseudo_label=arrays['teacher_pseudo_label'],
        s_phy_class_agreement=arrays['S_phy'], attenuation_phy=arrays['A_phy'],
        g_phy=arrays['G_phy'], w_before_phy=arrays['W_before'],
        w_after_phy=arrays['W_after'], delta_w_phy=arrays['W_after']-arrays['W_before'])
    maps['metadata'] = np.array(json.dumps(dict(iter=4000, sample_id='SYNTHETIC.tif',
        synthetic_fixture=True, original_shape=[256, 256], gate_min_weight=.75,
        input_view=['R', 'G', 'B'], capture_mode='online_after_train_iter',
        modules=dict(phy=True, struct=False, proto=False, edge=False), states=dict(phy='active'))))
    return maps


class OnlinePanelTests(unittest.TestCase):
    def test_horizontal_bars_and_pixel_exact_full256_center196_pairs(self):
        maps = online_maps()
        original = copy.deepcopy(maps)
        figure = plot.online_plate([maps], 'phys')
        try:
            images = [axis for axis in figure.axes if axis.images]
            self.assertEqual(len(images), 4)
            for axis, limits in zip(images[1:], ((0, 1), (0, 1), (0, .25))):
                self.assertEqual(axis.images[0].get_clim(), limits)
                bar = axis.images[0].colorbar
                self.assertEqual(bar.orientation, 'horizontal')
                self.assertLess(bar.ax.get_position().y1, axis.get_position().y0)
                self.assertAlmostEqual(bar.ax.get_position().width, axis.get_position().width)
                np.testing.assert_allclose(axis.images[0].get_cmap().get_bad(),
                                           [208/255, 208/255, 208/255, 1])
            scope = maps['valid_mask'] & np.isin(maps['pseudo_label'], [0, 1])
            np.testing.assert_array_equal(np.ma.getmaskarray(images[2].images[0].get_array()), ~scope)
            np.testing.assert_array_equal(np.ma.getmaskarray(images[3].images[0].get_array()),
                                          ~(scope & (maps['w_before_phy'] > 0)))
        finally:
            plot.plt.close(figure)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            output = Path(temporary)
            plot.render_online_node(maps, output, 4000)
            for key in ('target_image', 'w_conf', 's_phy_class_agreement',
                        'attenuation_phy', 'w_before_phy', 'w_after_phy', 'delta_w_phy'):
                with Image.open(output/'full256/iter_004000'/(key+'.png')) as full, \
                        Image.open(output/'center196/iter_004000'/(key+'.png')) as center:
                    self.assertEqual(full.size, (256, 256))
                    self.assertEqual(center.size, (196, 196))
                    np.testing.assert_array_equal(np.asarray(center), np.asarray(full)[30:226, 30:226])
            with Image.open(output/'full256/iter_004000/attenuation_phy.png') as full:
                np.testing.assert_array_equal(np.asarray(full)[0, 0], [208, 208, 208, 255])
            metadata = json.loads((output/'center196/iter_004000/metadata.json').read_text())
            self.assertEqual(metadata['export_shape'], [196, 196])
            self.assertEqual(metadata['display_crop_xyxy'], [30, 30, 226, 226])
            self.assertEqual(metadata['spatial_styles']['attenuation_phy']['vmax'], .25)
            for key in maps:
                np.testing.assert_array_equal(maps[key], original[key])


@unittest.skipIf(torch is None, 'PyTorch is required')
class OnlineSixImageTests(unittest.TestCase):
    def test_live_model_upgrade_six_samples_resume_and_readonly_state(self):
        from test_doris_mechanism import MechanismTests
        from test_hphy import HPhyTests
        HPhyTests.setUpClass()
        scope = HPhyTests.scope
        spec = importlib.util.spec_from_file_location('hphy_online_observation', ROOT/'mmseg/utils/mechanism.py')
        observation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observation)
        model = MechanismTests().model()
        model.phy_mode = 'hierarchical_gv'
        model.phy_cfg.update(mode='hierarchical_gv', phy_min_weight=.75,
            phy_gate_schedule='early_nonzero_linear', phy_gate_initial_strength=.10, phy_gate_full_iter=4000)
        model.phy_min_weight = .75
        model.phy_gate_params = scope['phy_gate_parameters'](model.phy_cfg)
        model.phy_head = scope['HierarchicalPhyReliabilityHead']()
        model.enable_struct = model.enable_proto = model.enable_struct_edge = False
        model.mechanism_cfg.update(probe_enabled=True, online_plot=True, strict=True,
                                   save_raw=True, make_figures=True, output_mode='branches')
        with torch.no_grad():
            model.ema_model.classifier.weight.zero_()
            model.ema_model.classifier.bias.fill_(-8)
            model.ema_model.classifier.bias[0] = 8
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        sum(parameter.sum() for parameter in model.parameters()).backward()
        optimizer.step()
        state = copy.deepcopy(model.state_dict())
        gradients = {name: value.grad.clone() for name, value in model.named_parameters() if value.grad is not None}
        modes = [module.training for module in model.modules()]
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        config = static_config('configs/hrda/doris/goji26RGB2goji30RGB/'
            'doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py')
        filenames = config['uda']['fixed_probe_filenames']

        def load_sample(probe, unused_model):
            index = filenames.index(probe['filename'])
            image = torch.full((1, 6, 256, 256), 30.+index)
            vi = torch.full((1, 3, 256, 256), .1*(index+1))
            metas = [dict(filename=probe['expected_path'], img_norm_cfg=dict(mean=[0]*6, std=[255]*6))]
            description = dict(probe, tensor_sha256=hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                               target_path=probe['expected_path'], original_shape=[256, 256])
            return (image, vi, metas), description

        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary:
            folder = Path(temporary)
            config['data']['train']['target']['data_root'] = str(folder/'30')
            config['data']['val']['data_root'] = str(folder/'30')
            previous = copy.deepcopy(config)
            previous['uda'].pop('fixed_probe_filenames')
            previous['uda']['fixed_probe_max_samples'] = 1
            runner = SimpleNamespace(work_dir=str(folder), iter=499, mechanism_config=previous,
                                     meta=dict(git_commit='synthetic'), logger=Mock())
            with patch.object(observation, 'load_probe_sample', side_effect=load_sample):
                old = observation.MechanismCollector(runner, model)
                model.local_iter = 500
                old.snapshot(500)
                previous_files = {path: path.read_bytes() for path in old.root.rglob('*') if path.is_file()}
                runner.iter = 500
                runner.resume_from = 'iter_500.pth'
                runner.mechanism_config = config
                upgraded = observation.MechanismCollector(runner, model)
                self.assertEqual(upgraded.identity, old.identity)
                self.assertEqual(upgraded.root, old.root)
                self.assertEqual(len(upgraded.extra_probes), 5)
                model.local_iter = 1000
                rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
                upgraded.snapshot(1000)
                for probe, filename in zip([upgraded]+upgraded.extra_probes, filenames):
                    self.assertEqual(probe.probe_cfg['filename'], filename)
                    source = probe.root/'fixed_probe/selected_arrays/iter_001000.npz'
                    with np.load(source) as maps:
                        self.assertEqual(json.loads(str(maps['metadata']))['sample_id'], filename)
                        self.assertEqual(maps['valid_mask'].shape, (256, 256))
                        np.testing.assert_allclose(maps['vi_ndvi'], .1*(filenames.index(filename)+1))
                    self.assertTrue((probe.root/'fixed_probe/snapshots/full256/iter_001000/attenuation_phy.png').is_file())
                    self.assertTrue((probe.root/'fixed_probe/snapshots/center196/iter_001000/attenuation_phy.png').is_file())
                    self.assertEqual(set(probe.manifest['snapshots']['1000']['native_panels']),
                                     {'full256', 'center196'})
                    if probe is not upgraded:
                        self.assertFalse((probe.root/'fixed_probe/selected_arrays/iter_000500.npz').exists())
                for path, content in previous_files.items():
                    if path.name not in ('manifest.json', 'probe_cost.csv'):
                        self.assertEqual(path.read_bytes(), content)
                self.assertEqual(random.getstate(), rng[0])
                np.testing.assert_equal(np.random.get_state(), rng[1])
                torch.testing.assert_close(torch.get_rng_state(), rng[2], rtol=0, atol=0)
                with patch.object(observation, 'probe_arrays', side_effect=AssertionError('Repeated inference')):
                    upgraded.snapshot(1000)
                runner.iter = 1000
                runner.resume_from = 'iter_1000.pth'
                resumed = observation.MechanismCollector(runner, model)
                self.assertEqual(len(resumed.extra_probes), 5)
                self.assertEqual(resumed.root, upgraded.root)
                for probe, original in zip(resumed.extra_probes, upgraded.extra_probes):
                    self.assertEqual(probe.root, original.root)
        self.assertEqual(modes, [module.training for module in model.modules()])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        for name, value in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(value.grad, gradients[name], rtol=0, atol=0)
        for key, values in optimizer_state['state'].items():
            for field, value in values.items():
                torch.testing.assert_close(value, optimizer.state_dict()['state'][key][field], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
