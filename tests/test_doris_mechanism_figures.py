"""Forty synthetic nodes validate exports; they are not scientific results."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class MechanismFigureTests(unittest.TestCase):
    def test_process_curves_omit_disabled_modules_instead_of_filling_zeros(self):
        spec = importlib.util.spec_from_file_location('process_curves',
            ROOT / 'tools/plot_goji_e1_process_curves.py')
        plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            root = Path(folder)
            path = root / 'mechanism.csv'
            row = dict(iter=500, data_source='training_batch_actual', w_conf_mean_valid='.9',
                w_t_ps_mean_valid='', w_t_phy_mean_valid='', w_t_proto_mean_valid='.7',
                proto_target_update_ratio='.2', boundary_weight_mean_valid='', struct_enabled='0')
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader(); writer.writerows([row, dict(row, iter=1000)])
            rows = plot.read_rows(path)
            self.assertNotIn('mparc_pseudo_weight', rows[0])
            self.assertNotIn('boundary_weight_x100', rows[0])
            self.assertNotIn('physiology_pseudo_weight', rows[0])
            self.assertEqual(rows[0]['prototype_pseudo_weight'], .7)
            self.assertEqual(len(plot.plot_curves(rows, root, 'synthetic', ['png'], 60)), 1)
            self.assertEqual(len(plot.plot_two_panel_curves(rows, root, 'synthetic', ['png'], 60)), 1)
            self.assertEqual(plot.plt.get_fignums(), [])

    def test_individual_modules_and_combinations_render_only_their_evidence(self):
        spec = importlib.util.spec_from_file_location('module_figures',
            ROOT / 'tools/build_doris_mechanism_figures.py')
        plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot)
        from tools.check_doris_stage1_configs import VARIANTS
        for suffix, flags in VARIANTS[1:]:
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
                root = Path(folder)
                arrays = root / 'fixed_probe/arrays'
                arrays.mkdir(parents=True)
                (root / 'metrics').mkdir()
                enabled = dict(zip(('phy', 'struct', 'proto', 'edge'), flags))
                manifest = dict(modules=enabled, proto_weight_start=12000 if flags[2] else None,
                    proto_target_update_start=8000 if flags[2] else None,
                    phy_warmup_start=6000 if flags[0] else None,
                    struct_warmup_start=6000 if flags[1] else None)
                (root / 'manifest.json').write_text(json.dumps(manifest))
                rows = []
                for step in (500, 5500, 6000, 6500, 11500, 12000, 20000):
                    base = np.full((8, 8), .8, np.float32)
                    maps = dict(w_base=base, w_conf=base, valid_mask=np.ones((8, 8), bool),
                        target_image=np.ones((8, 8, 3), np.float32) * .5,
                        proto_active=np.array(flags[2] and step >= 12000))
                    ps = base * (.9 if step > 6000 and (flags[0] or flags[1]) else 1)
                    if flags[0]:
                        maps.update(r_phy=base, w_t_phy=ps, delta_w_phy=ps-base)
                    if flags[1]:
                        maps.update({key: base for key in
                            ('r_conf', 'r_loc', 'r_adj', 'r_prob', 'r_struct')})
                        maps.update(adjacency_available=maps['valid_mask'],
                                    semantic_available=maps['valid_mask'])
                    if flags[0] or flags[1]:
                        maps.update(w_t_ps=ps, delta_w_ps=ps-base)
                    if flags[2]:
                        final = ps * (.9 if step >= 12000 else 1)
                        maps.update(w_pre_proto=ps, w_t_proto=final, delta_w_proto=final-ps)
                        for cls in range(5):
                            rows.append(dict(iter=step, class_id=cls, class_name=str(cls),
                                data_source='training_batch_actual', tau_proto=.25, tau_conf_proto=.88,
                                update_active=int(step >= 8000), predicted_count=20,
                                confidence_pass_count=12, joint_selected_count=8, n_sampled_batches=5))
                    if flags[3]:
                        maps.update({key: base for key in
                            ('e_prob', 'e_img', 'r_pair', 'r_edge', 'boundary_weight')})
                    maps['metadata'] = np.array(json.dumps(dict(iter=step, sample_id='SYNTHETIC_MODULE',
                        sample_tensor_sha256='fixture', config_sha256='fixture',
                        data_source='fixed_probe_recomputed', modules=enabled, synthetic_fixture=True)))
                    np.savez_compressed(arrays / ('iter_{:06d}.npz'.format(step)), **maps)
                if rows:
                    with (root / 'metrics/proto_selection_500.csv').open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=rows[0])
                        writer.writeheader(); writer.writerows(rows)
                plot.render_node(maps, root / 'online', 20000)
                expected_online = (int(flags[0]) + int(flags[1]) +
                                   int(flags[1] or flags[2]) + int(flags[3]))
                self.assertEqual(len(list((root / 'online').glob('*.png'))), expected_online)
                output = plot.build_figures(root)
                expected_figures = int(flags[0]) + int(flags[1]) + int(flags[3]) + 1 + int(flags[2])
                self.assertEqual(len(list(output.glob('*.pdf'))), expected_figures)
                self.assertEqual((output / 'fig_proto_selection.png').exists(), flags[2])
                self.assertEqual((output / 'fig_struct_components.png').exists(), flags[1])
                self.assertEqual((output / 'fig_physiology.png').exists(), flags[0])
                self.assertEqual((output / 'fig_edge_components.png').exists(), flags[3])
                provenance = json.loads((output / 'figure_selection.json').read_text())
                self.assertEqual('training_batch_actual' in provenance['data_sources'], flags[2])
                if flags[2]:
                    self.assertIn(11500, provenance['weight_iterations'])
                    self.assertIn(12000, provenance['weight_iterations'])
                self.assertEqual(plot.plt.get_fignums(), [])

    def test_forty_node_workflow_generates_three_traceable_figures(self):
        spec = importlib.util.spec_from_file_location('mechanism_figures',
            ROOT / 'tools/build_doris_mechanism_figures.py')
        plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot)
        folder = Path(tempfile.mkdtemp(prefix='mechanism_figure_fixture_', dir=ROOT / 'outputs'))
        arrays = folder / 'fixed_probe/arrays'
        arrays.mkdir(parents=True)
        (folder / 'metrics').mkdir()
        (folder / 'manifest.json').write_text(json.dumps(dict(
            proto_weight_start=12000, proto_target_update_start=8000)))
        image = np.random.default_rng(15).random((24, 24, 3)).astype(np.float32)
        rows = []
        for step in range(500, 20001, 500):
            confidence = np.full((24, 24), 0.5 + step / 50000, np.float32)
            w_ps = confidence * 0.9
            active = step >= 12000
            final = w_ps * (0.95 if active else 1)
            metadata = dict(iter=step, sample_id='SYNTHETIC_TEST_SAMPLE',
                sample_tensor_sha256='fixed-fixture', config_sha256='fixture',
                state_sha256='synthetic-state-' + str(step),
                data_source='fixed_probe_recomputed', synthetic_fixture=True)
            maps = dict(target_image=image, valid_mask=np.ones((24, 24), bool),
                adjacency_available=np.full((24, 24), step >= 1000),
                semantic_available=np.full((24, 24), step >= 1000),
                r_conf=confidence, r_loc=confidence * .8, r_adj=confidence * .6,
                r_prob=confidence * .7, r_struct=confidence * .7,
                w_conf=confidence, w_t_ps=w_ps, w_t_proto=final,
                delta_w_proto=final - w_ps, proto_active=np.array(active),
                pseudo_label=np.zeros((24, 24), np.uint8),
                metadata=np.array(json.dumps(metadata)))
            np.savez_compressed(arrays / ('iter_{:06d}.npz'.format(step)), **maps)
            if step in (500, 1000):
                plot.render_node(maps, folder / 'fixed_probe/snapshots', step)
            for cls, name in enumerate(('goji', 'vegetation', 'building', 'water', 'road')):
                rows.append(dict(iter=step, class_id=cls, class_name=name,
                    data_source='training_batch_actual',
                    tau_proto=.25, tau_conf_proto=.88, update_active=int(step >= 8000),
                    predicted_count=100 + cls * 20, confidence_pass_count=50,
                    joint_selected_count=30, n_sampled_batches=5))
        path = folder / 'metrics/proto_selection_500.csv'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader(); writer.writerows(rows)
        output = plot.build_figures(folder)
        self.assertEqual(len(list(arrays.glob('*.npz'))), 40)
        self.assertEqual(len(list(output.glob('*.pdf'))), 3)
        self.assertEqual(len(list(output.glob('*.png'))), 3)
        provenance = json.loads((output / 'figure_selection.json').read_text())
        self.assertIn(11500, provenance['weight_iterations'])
        self.assertIn(12000, provenance['weight_iterations'])
        self.assertEqual(provenance['struct_iter'], 20000)
        self.assertTrue(provenance['synthetic_fixture'])
        for path in output.glob('*.png'):
            with Image.open(path) as img:
                self.assertGreaterEqual(img.info['dpi'][0], 299)
                self.assertIn('synthetic_fixture', img.info['Description'])
                img.verify()
        stamps = {p.name: p.stat().st_mtime_ns for p in output.glob('*.png')}
        plot.build_figures(folder)
        self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in output.glob('*.png')})
        self.assertEqual(plot.plt.get_fignums(), [])
        print('SYNTHETIC QA artifacts:', output)


if __name__ == '__main__':
    unittest.main()
