"""Structural Student paper panels and diagnostics on six synthetic cases."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from test_struct_mechanism import ROOT, fixture_arrays
from tools.doris_struct_evidence import CONTRIBUTIONS, prepare_struct_evidence, struct_statistics
from tools import struct_mechanism_panels as panels
from tools.build_doris_mechanism_figures import online_plate, render_online_node
from tools.build_struct_mechanism_figure import render


def online_payload(arrays, meta):
    result = {key: value.copy() for key, value in arrays.items()}
    result.update(target_image=arrays['image_rgb'].astype(np.float32)/255,
        pseudo_label=arrays['teacher_pseudo_label'], r_conf=arrays['R_conf'], r_loc=arrays['R_loc'],
        r_adj=arrays['R_adj'], r_prob=arrays['R_prob'], r_struct=arrays['R_struct'],
        w_before_struct=arrays['W_before'], w_after_struct=arrays['W_after'])
    info = dict(meta, iter=12000, struct_cfg=meta['gate'],
        modules=dict(phy=False, struct=True, proto=False, edge=False))
    result['metadata'] = np.array(json.dumps(info))
    return result


class StructuralVisualizationTests(unittest.TestCase):
    def test_six_cases_weights_masks_contributions_and_schedule(self):
        for index, (ramp, mode, weight) in enumerate(((0., 'soft', .8), (.125, 'soft', .8),
                (.5, 'soft', .8), (1., 'soft', .8), (1., 'hard', .8), (1., 'soft', 0.))):
            with self.subTest(index=index):
                arrays, meta = fixture_arrays(ramp, mode, weight)
                meta['gate']['rule_iter'] = int(6000+4000*ramp)
                meta['sample_id'] = 'SYNTHETIC_{}.tif'.format(index)
                original = copy.deepcopy(arrays)
                prepared, info = prepare_struct_evidence(arrays, meta)
                self.assertTrue(info['struct_validation']['passed'], info['struct_validation'])
                mask = prepared['A_struct_valid_mask']
                expected = arrays['valid_mask'] & np.isin(arrays['teacher_pseudo_label'], [2, 3, 4]) & (arrays['W_before'] > 0)
                np.testing.assert_array_equal(mask, expected)
                np.testing.assert_array_equal(prepared['struct_display_mask'],
                    arrays['valid_mask'] & np.isin(arrays['teacher_pseudo_label'], [2, 3, 4]))
                np.testing.assert_allclose(prepared['W_after'][mask],
                    prepared['W_before'][mask]*(1-prepared['A_struct'][mask]), atol=1e-12)
                np.testing.assert_allclose(sum(prepared[key] for key in CONTRIBUTIONS), prepared['R_struct'], atol=1e-6)
                for key in original:
                    np.testing.assert_array_equal(arrays[key], original[key])
                stats = struct_statistics(prepared, info)
                self.assertEqual(stats['attenuation_by_class'][-1]['struct_active_pixel_count'], int(mask.sum()))
                if weight == 0:
                    self.assertIsNone(stats['attenuation_by_class'][-1]['A_struct_mean'])
                    self.assertFalse(mask.any())
                if ramp == 0:
                    self.assertEqual(stats['attenuation_by_class'][-1]['struct_changed_pixel_count'], 0)
                if mode == 'soft' and ramp == 1:
                    self.assertAlmostEqual(info['attenuation_color_range'][1], .675)

    def test_anomalies_are_recorded_and_never_clipped(self):
        for issue in ('increase', 'negative', 'nonfinite'):
            arrays, meta = fixture_arrays()
            if issue == 'increase':
                arrays['W_after'][80, 130] = arrays['W_before'][80, 130]*1.2
            elif issue == 'negative':
                arrays['W_after'][80, 130] = -.1
            else:
                arrays['W_after'][80, 130] = np.inf
            prepared, info = prepare_struct_evidence(arrays, meta)
            self.assertFalse(info['struct_validation']['passed'])
            if issue == 'increase':
                self.assertLess(prepared['A_struct'][80, 130], 0)
            with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary:
                out = Path(temporary)/'invalid'
                with self.assertRaisesRegex(ValueError, 'audit saved'):
                    panels.render_struct_bundle(arrays, out, meta, dpi=30)
                self.assertTrue((out/'validation.json').is_file())
                self.assertTrue((out/'responses_full_256.npz').is_file())
                self.assertFalse((out/'struct_full_raw.png').exists())

    def test_two_grays_fixed_attenuation_scale_contrast_and_layout(self):
        arrays, meta = fixture_arrays()
        prepared, info = prepare_struct_evidence(arrays, meta)
        center = panels.crop_arrays(prepared)
        full = panels.display_scales(center, info, 'full')
        enhanced = panels.display_scales(center, info, 'contrast')
        self.assertEqual(full['A_struct'], enhanced['A_struct'])
        self.assertNotEqual(full['R_loc'], enhanced['R_loc'])
        pixels = panels.pixels_for(prepared, 'R_loc', 'struct', full)
        np.testing.assert_array_equal(pixels[0, 0], panels.INVALID)
        np.testing.assert_array_equal(pixels[80, 40], panels.BACKGROUND)
        for mode in ('formal', 'full', 'struct', 'contrast'):
            scales = panels.display_scales(center, info, mode)
            fig = panels.panel_figure(center, panels.COMPONENTS, mode, scales, info)
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                axes = [axis for axis in fig.axes if axis.images]
                self.assertEqual(len(axes), 7)
                self.assertEqual(panels.COMPONENTS[-1], 'struct_prediction')
                self.assertNotIn('A_struct', panels.COMPONENTS)
                self.assertIn('Student', axes[-1].get_title())
                np.testing.assert_array_equal(axes[-1].images[0].get_array(),
                    panels.pixels_for(center, 'struct_prediction', 'full', full))
                for axis in axes:
                    bounds = axis.title.get_window_extent(renderer)
                    self.assertLessEqual(bounds.y1, fig.bbox.y1)
                    self.assertGreaterEqual(bounds.x0, 0)
                    self.assertLessEqual(bounds.x1, fig.bbox.x1)
                for left, right in zip(axes, axes[1:]):
                    self.assertFalse(left.title.get_window_extent(renderer).overlaps(right.title.get_window_extent(renderer)))
                bars = sorted([axis for axis in fig.axes if not axis.images], key=lambda axis: axis.get_position().x0)
                for left, right in zip(bars, bars[1:]):
                    self.assertFalse(left.get_tightbbox(renderer).overlaps(right.get_tightbbox(renderer)))
            finally:
                panels.plt.close(fig)
        fig = online_plate([online_payload(arrays, meta)], 'struct')
        self.assertEqual(len([axis for axis in fig.axes if axis.images]), 7)
        panels.plt.close(fig)

    def test_online_and_checkpoint_rendering_identical_pixels_and_statistics(self):
        arrays, meta = fixture_arrays(gt=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary:
            root = Path(temporary)
            source = root/'responses.npz'
            np.savez_compressed(source, **arrays)
            meta['arrays_sha256'] = {'responses_full_256.npz': panels.file_sha256(source)}
            metadata = root/'meta.json'
            metadata.write_text(json.dumps(meta), encoding='utf-8')
            # Keep this test focused on the two production entries and representative exports.
            original_export = panels.export_single
            def representative_export(current, key, mode, scales, info, folder, dpi, font_size):
                if key in ('A_struct', 'struct_prediction'):
                    original_export(current, key, mode, scales, info, folder, dpi, font_size)
            with patch.object(panels, 'export_single', side_effect=representative_export):
                offline = render(source, metadata, root/'offline', dpi=35)
                paths = render_online_node(online_payload(arrays, meta), root/'online', 12000, dpi=35)
            online = paths[0].parent
            for name in ('struct_formal_group_1x7.png', 'r_struct_struct_only_formal.png',
                         'struct_student_prediction_formal.png', 'struct_full_raw.png', 'struct_only_raw.png', 'struct_only_contrast.png',
                         'struct_contrib.png', 'panels_196/A_struct_no_title.png',
                         'panels_196/struct_prediction_no_title.png'):
                with Image.open(offline/name) as left, Image.open(online/name) as right:
                    np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
            left = json.loads((offline/'struct_stats.json').read_text())
            right = json.loads((online/'struct_stats.json').read_text())
            self.assertEqual(left, right)
            settings = json.loads((offline/'render_meta.json').read_text())
            self.assertEqual(settings['formal_prediction_identity'], 'Student test-mode prediction')
            self.assertEqual(settings['formal_group_panels'][-1], 'struct_prediction')
            self.assertEqual(settings['formal_candidate_figures'], ['struct_formal_group_1x7'])
            self.assertEqual(settings['formal_response_range'], [0., 1.])
            self.assertEqual(set(settings['render']['scales'][panels.FORMAL_NAME]),
                {'R_conf', 'R_loc', 'R_adj', 'R_prob', 'R_struct'})
            self.assertTrue(all(bounds == [0., 1.] for bounds in settings['render']['scales'][panels.FORMAL_NAME].values()))
            self.assertEqual(settings['formal_mask_source'], 'EMA Teacher class IDs')
            self.assertTrue(settings['formal_version'])
            self.assertEqual(panels.file_sha256(source), meta['arrays_sha256']['responses_full_256.npz'])
            # Same provenance can be resumed without overwriting any file.
            before = {p: p.stat().st_mtime_ns for p in online.rglob('*') if p.is_file()}
            render_online_node(online_payload(arrays, meta), root/'online', 12000, dpi=35)
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in online.rglob('*') if p.is_file()})

    def test_cpu_cli_uses_old_weights_and_explicit_resolved_configuration(self):
        arrays, meta = fixture_arrays()
        maps = online_payload(arrays, meta)
        old_info = json.loads(str(maps['metadata']))
        old_info.pop('struct_cfg')
        old_info.pop('gate')
        maps['metadata'] = np.array(json.dumps(old_info))
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary:
            root = Path(temporary)
            source = root/'iter_012000.npz'
            np.savez_compressed(source, **maps)
            config = root/'saved_training.py'
            config.write_text('uda = '+repr(dict(struct_cfg=meta['gate'])), encoding='utf-8')
            # Exercise real argument parsing, config loading and CPU derivation; export I/O is covered above.
            from tools import build_struct_mechanism_figure as builder
            argv = ['build_struct_mechanism_figure.py', '--input-npz', str(source), '--config', str(config),
                    '--output-dir', str(root/'redraw'), '--dpi', '30']
            with patch.object(sys, 'argv', argv), patch.object(panels, 'export_single'):
                builder.main()
            self.assertTrue(list((root/'redraw').glob('*/struct_formal_group_1x7.png')))
            self.assertTrue((root/'redraw/struct_samples_summary.csv').is_file())
            self.assertEqual(np.load(source)['w_before_struct'].shape, (256, 256))

    def test_formal_mask_uses_student_palette_with_all_classes_and_exact_crop(self):
        arrays, meta = fixture_arrays()
        arrays['struct_prediction'] = np.roll(arrays['teacher_pseudo_label'], 35, axis=1)
        prepared, info = prepare_struct_evidence(arrays, meta)
        center = panels.crop_arrays(prepared)
        pixels = panels.pixels_for(center, 'struct_prediction', 'struct', panels.display_scales(center, info, 'struct'))
        for cls, color in enumerate(panels.PALETTE):
            mask = center['struct_prediction'] == cls
            self.assertTrue(mask.any())
            np.testing.assert_array_equal(pixels[mask], np.broadcast_to(color, pixels[mask].shape))
        np.testing.assert_array_equal(center['struct_prediction'], arrays['struct_prediction'][30:226, 30:226])
        self.assertFalse(np.array_equal(center['struct_prediction'], center['teacher_pred']))
        del arrays['struct_prediction']
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary:
            with self.assertRaisesRegex(ValueError, 'requires struct_prediction'):
                panels.render_struct_bundle(arrays, Path(temporary)/'formal', meta, dpi=30)

    def test_formal_responses_use_teacher_mask_and_fixed_scale_without_data_changes(self):
        arrays, meta = fixture_arrays(gt=True)
        arrays['struct_prediction'] = np.roll(arrays['teacher_pseudo_label'], 35, axis=1)
        original = copy.deepcopy(arrays)
        prepared, info = prepare_struct_evidence(arrays, meta)
        center = panels.crop_arrays(prepared)
        scales = panels.display_scales(center, info, 'formal')
        expected_mask = center['valid_mask'] & np.isin(center['teacher_pseudo_label'], [2, 3, 4])
        for key in ('R_conf', 'R_loc', 'R_adj', 'R_prob', 'R_struct'):
            self.assertEqual(scales[key], [0., 1.])
            np.testing.assert_array_equal(prepared[key], original[key])
            expected = expected_mask.copy()
            if key in ('R_adj', 'R_prob'):
                expected &= center['adjacency_available' if key == 'R_adj' else 'semantic_available']
            np.testing.assert_array_equal(panels.mask_for(center, key, 'formal'), expected)
            pixels = panels.pixels_for(center, key, 'formal', scales)
            other_classes = center['valid_mask'] & ~expected_mask
            np.testing.assert_array_equal(pixels[other_classes], np.broadcast_to(panels.BACKGROUND, pixels[other_classes].shape))
        np.testing.assert_array_equal(panels.pixels_for(center, 'image_rgb', 'formal', scales), center['image_rgb'])
        self.assertFalse(np.array_equal(expected_mask, np.isin(center['struct_prediction'], [2, 3, 4])))
        self.assertFalse(np.array_equal(expected_mask, np.isin(center['gt_label'], [2, 3, 4])))


if __name__ == '__main__':
    unittest.main()
