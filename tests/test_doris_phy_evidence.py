"""Offline class-reference and aligned physiology figure tests using synthetic responses."""
import ast
import csv
import hashlib
import importlib.util
import io
import json
import os.path as osp
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from tools.doris_phy_evidence import add_reference_evidence, load_phy_reference, class_statistics_csv

ROOT = Path(__file__).resolve().parents[1]
meta = {}
exec((ROOT/'mmseg/datasets/goji_meta.py').read_text(), meta)
ANNOTATION = dict(class_names=list(meta['GOJI_CLASSES']), palette=meta['GOJI_PALETTE'])


def fixture():
    reference = np.tile(np.repeat(np.arange(3,dtype=np.uint8),32), (64,1))
    response = np.tile(np.linspace(0,1,96,dtype=np.float32), (64,1))
    before = np.full(reference.shape,.7,np.float32)
    pseudo = reference.copy()
    pseudo[:,64:] = 0
    full_gate = np.where(response < .3, (.82+.18*response)*.5, .82+.18*response)
    after = before*(.75+.25*full_gate)
    maps = dict(target_image=np.dstack([response,response*.8,response*.6]),
        valid_mask=np.ones(reference.shape,bool), pseudo_label=pseudo,
        w_conf=.55+.4*response, r_phy=response, w_before_phy=before,
        w_after_phy=after, delta_w_phy=after-before, m_phy=np.isin(pseudo,[0,1]))
    maps['metadata'] = np.array(json.dumps(dict(iter=7000,sample_id='SYNTHETIC.tif',
        original_shape=list(reference.shape),synthetic_fixture=True,
        modules=dict(phy=True,struct=False,proto=False,edge=False),states=dict(phy='ramping'),
        capture_mode='checkpoint_replay',data_source='checkpoint_replay')))
    return maps, reference


def annotation_loader():
    tree = ast.parse((ROOT/'mmseg/datasets/pipelines/loading.py').read_text())
    node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LoadAnnotations')
    node.decorator_list = []
    mmcv = SimpleNamespace(FileClient=lambda **kwargs:SimpleNamespace(get=lambda path:Path(path).read_bytes()),
        imfrombytes=lambda data,**kwargs:np.array(Image.open(io.BytesIO(data))))
    scope = dict(np=np,osp=osp,mmcv=mmcv)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'existing_annotation_loader','exec'),scope)
    return scope['LoadAnnotations']


class PhysiologyReferenceTests(unittest.TestCase):
    def test_class_summaries_retain_response_values_and_mark_absent_classes(self):
        maps, reference = fixture()
        before = {key:value.copy() for key,value in maps.items() if key!='metadata'}
        maps['valid_mask'][0] = False
        before['valid_mask'] = maps['valid_mask'].copy()
        add_reference_evidence(maps,reference,ANNOTATION)
        evidence = json.loads(str(maps['metadata']))['phy_evidence']
        self.assertEqual(evidence['classes'][0]['valid_pixels'],63*32)
        self.assertEqual(evidence['classes'][2]['teacher_correct_fraction'],0)
        self.assertEqual(evidence['classes'][3]['valid_pixels'],0)
        self.assertIsNone(evidence['classes'][3]['r_phy_mean'])
        for key,value in before.items():np.testing.assert_array_equal(maps[key],value)
        rows = list(csv.DictReader(io.StringIO(class_statistics_csv(evidence))))
        self.assertEqual(len(rows),5)
        self.assertIn('w_conf_mean',rows[0])

    def test_native_label_mapping_and_ignore_index_are_reused(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root = Path(directory)
            path = root/'ann_dir/val/SYNTHETIC.tif'
            path.parent.mkdir(parents=True)
            raw = np.tile(np.arange(6,dtype=np.uint8),(64,16))
            Image.fromarray(raw).save(path)
            maps,_ = fixture()
            config = dict(data=dict(val=dict(type='GojiDataset',data_root=str(root),ann_dir='ann_dir/val')))
            modules = {'mmseg.datasets.pipelines':SimpleNamespace(LoadAnnotations=annotation_loader()),
                       'mmseg.datasets.goji_meta':SimpleNamespace(
                           GOJI_CLASSES=ANNOTATION['class_names'],GOJI_PALETTE=ANNOTATION['palette'])}
            with patch.dict('sys.modules',modules):load_phy_reference(maps,config)
            expected = np.where(raw==0,255,raw.astype(np.int16)-1).astype(np.uint8)
            np.testing.assert_array_equal(maps['reference_label'],expected)
            with self.assertRaisesRegex(ValueError,'evaluation mapping'):
                add_reference_evidence(maps,np.full(expected.shape,8),ANNOTATION)


class PhysiologySemanticTests(unittest.TestCase):
    def test_goji_vegetation_confusion_is_compared_within_each_true_class(self):
        from tools.check_doris_phy_semantics_7000 import semantic_statistics, vegetation_confusion_map
        maps, reference = fixture()
        maps['pseudo_label'][:,16:32] = 1
        maps['pseudo_label'][:,48:64] = 0
        response = np.full(reference.shape,.9,np.float32)
        response[:,:16] = .4
        response[:,16:32] = .8
        response[:,48:64] = .6
        maps['r_phy'] = response
        maps['w_after_phy'] = maps['w_before_phy']*(.75+.25*(.82+.18*response))
        maps['delta_w_phy'] = maps['w_after_phy']-maps['w_before_phy']
        add_reference_evidence(maps,reference,ANNOTATION)
        result = semantic_statistics(maps)['vegetation_class_check']
        self.assertEqual([row['valid_pixels'] for row in result['groups']],[1024]*4)
        self.assertEqual(result['status'],'partially_supported')
        self.assertGreater(result['comparisons'][0]['delta_difference_error_minus_correct'],0)
        self.assertLess(result['comparisons'][1]['delta_difference_error_minus_correct'],0)
        groups = vegetation_confusion_map(maps)
        for start,stop,code in ((0,16,1),(16,32,2),(32,48,4),(48,64,3),(64,96,0)):
            self.assertTrue((groups[:,start:stop]==code).all())
        maps['valid_mask'][0] = False
        reference[1,0] = 255
        maps['reference_label'] = reference
        groups = vegetation_confusion_map(maps)
        self.assertTrue((groups[0]==0).all())
        self.assertEqual(groups[1,0],0)

    def test_comparison_excludes_margins_ignore_and_nonvegetation_predictions(self):
        from tools.check_doris_phy_semantics_7000 import semantic_statistics
        maps, reference = fixture()
        maps['valid_mask'][0] = False
        reference[1, 0] = 255
        maps['pseudo_label'][1, 1] = 2
        maps['m_phy'] = np.isin(maps['pseudo_label'], [0, 1])
        add_reference_evidence(maps, reference, ANNOTATION)
        result = semantic_statistics(maps)
        self.assertEqual(result['groups'][0]['valid_pixels'], 63*64-2)
        self.assertEqual(result['groups'][1]['valid_pixels'], 63*32)
        self.assertEqual(result['excluded_ignore_pixels'], 1)
        self.assertEqual(result['comparison']['status'], 'expected_direction_not_observed')
        maps['m_phy'][2, 0] = False
        with self.assertRaisesRegex(ValueError, 'physiology mask'):
            semantic_statistics(maps)

    def test_supporting_direction_and_missing_group_are_reported(self):
        from tools.check_doris_phy_semantics_7000 import semantic_statistics
        maps, reference = fixture()
        maps['r_phy'] = np.where(reference == 2, .1, .9).astype(np.float32)
        gate = .82+.18*maps['r_phy']
        gate = np.where(maps['r_phy'] < .3, gate*.5, gate)
        maps['w_after_phy'] = maps['w_before_phy']*(.75+.25*gate)
        maps['delta_w_phy'] = maps['w_after_phy']-maps['w_before_phy']
        add_reference_evidence(maps, reference, ANNOTATION)
        result = semantic_statistics(maps)
        self.assertEqual(result['comparison']['status'], 'consistent_direction')
        self.assertLess(result['comparison']['response_difference_false_minus_true'], 0)
        self.assertLess(result['comparison']['delta_difference_false_minus_true'], 0)
        maps['pseudo_label'][reference == 2] = 2
        maps['m_phy'] = np.isin(maps['pseudo_label'], [0, 1])
        result = semantic_statistics(maps)
        self.assertEqual(result['comparison']['status'], 'insufficient_groups')
        self.assertIsNone(result['groups'][1]['r_phy_mean'])
        metadata = json.loads(str(maps['metadata']))
        metadata['iter'] = 10000
        maps['metadata'] = np.array(json.dumps(metadata))
        self.assertEqual(semantic_statistics(maps)['iter'], 10000)

    def test_exports_are_repeatable_and_source_is_preserved(self):
        from tools.check_doris_phy_semantics_7000 import export_semantic_check
        maps, reference = fixture()
        add_reference_evidence(maps, reference, ANNOTATION)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            source = Path(directory)/'responses.npz'
            np.savez_compressed(source, **maps)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            output, result = export_semantic_check(source)
            repeated, _ = export_semantic_check(source)
            self.assertEqual(output, repeated)
            self.assertEqual(result['source_npz_sha256'], digest)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            rows = list(csv.DictReader(io.StringIO((output/'semantic_statistics.csv').read_text(encoding='utf-8'))))
            self.assertEqual(len(rows), 4)
            self.assertIn('缺少', (output/'semantic_summary.txt').read_text(encoding='utf-8'))


@unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Matplotlib unavailable')
class PhysiologyFigureTests(unittest.TestCase):
    def paper_fixture(self):
        y, x = np.mgrid[:256,:256].astype(np.float32)/255
        valid = np.zeros((256,256),bool)
        valid[30:226,30:226] = True
        before = np.full(valid.shape,.55,np.float32)
        after = before-.08*x
        maps = dict(target_image=np.dstack([x,y,x*y]),valid_mask=valid,
            w_conf=.5+.5*x,r_phy=x,w_before_phy=before,w_after_phy=after,delta_w_phy=after-before,
            pseudo_label=(x*4).astype(np.uint8),reference_label=(y*4).astype(np.uint8),
            m_phy=(x<.5))
        maps['metadata'] = np.array(json.dumps(dict(iter=7000,sample_id='SYNTHETIC.tif',
            synthetic_fixture=True,original_shape=[256,256],capture_mode='checkpoint_replay',
            phy_evidence=ANNOTATION)))
        return maps

    def test_paper_crop_uses_the_same_pixels_and_explicit_shared_scales(self):
        from tools.build_doris_mechanism_figures import phy_paper_plate, plt
        maps = self.paper_fixture()
        original = {key:value.copy() for key,value in maps.items()}
        fig,cropped,settings = phy_paper_plate(maps)
        try:
            self.assertEqual(settings['crop_xyxy'],[30,30,226,226])
            self.assertEqual(settings['export_shape'],[196,196])
            axes = [axis for axis in fig.axes if axis.images]
            self.assertEqual(len(axes),5)
            for axis,key in zip(axes,settings['panel_keys']):
                if key == 'phy_semantic_check':
                    from tools.check_doris_phy_semantics_7000 import vegetation_confusion_map, SEMANTIC_PALETTE
                    expected = np.asarray(SEMANTIC_PALETTE,np.uint8)[vegetation_confusion_map(maps)[30:226,30:226]]
                else:
                    expected = maps[key][30:226,30:226]
                np.testing.assert_array_equal(axis.images[0].get_array(),expected)
            self.assertEqual(axes[3].images[0].get_clim(),(-.1,0))
            self.assertEqual([axis.get_title(loc='left')[:3] for axis in axes],['(a)','(b)','(c)','(d)','(e)'])
            self.assertEqual(settings['layout'],'1x5')
            self.assertAlmostEqual(settings['weight_statistics']['pre_gate_weight_mean'],.55,places=6)
            self.assertEqual(settings['weight_statistics']['pre_gate_weight_std'],0)
            captions = '\n'.join(text.get_text() for text in fig.texts)
            for label in ('Pre-gate weight mean','after-gate weight mean','Mean ΔW','maximum decrease','changed valid pixels'):
                self.assertIn(label,captions)
            self.assertEqual(len(axes[0].patches),0)
            for key,value in original.items():np.testing.assert_array_equal(maps[key],value)
        finally:plt.close(fig)

    def test_paper_export_preserves_source_and_retains_crop_provenance(self):
        from tools.build_doris_mechanism_figures import render_phy_paper
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root = Path(directory)
            source = root/'responses.npz'
            np.savez_compressed(source,**self.paper_fixture())
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            original = root/'iter_007000_phys.png'
            original.write_bytes(b'preserved original figure fixture')
            files = render_phy_paper(source)
            self.assertTrue(all(path.is_file() for path in files))
            with Image.open(root/'paper/spatial/target_image.png') as image:
                self.assertEqual(image.size,(196,196))
                expected = (self.paper_fixture()['target_image'][30:226,30:226]*255).round().astype(np.uint8)
                np.testing.assert_array_equal(np.array(image),expected)
            for path in (root/'paper/spatial').glob('*.png'):
                with Image.open(path) as image:self.assertEqual(image.size,(196,196),path.name)
            for name in ('pseudo_label','reference_label'):
                with Image.open(root/'paper/spatial'/(name+'.png')) as image:
                    labels = self.paper_fixture()[name][30:226,30:226]
                    np.testing.assert_array_equal(np.array(image),np.asarray(ANNOTATION['palette'],np.uint8)[labels])
            with Image.open(root/'paper/spatial/valid_mask.png') as image:
                np.testing.assert_array_equal(np.array(image),np.full((196,196),255,np.uint8))
            with Image.open(root/'paper/spatial/target_image.tif') as image:
                self.assertEqual(image.size,(196,196))
                np.testing.assert_array_equal(np.array(image),expected)
            metadata = json.loads((root/'paper/iter_007000_phys_paper_metadata.json').read_text())
            self.assertEqual(metadata['display']['crop_xyxy'],[30,30,226,226])
            self.assertEqual(metadata['source_npz_sha256'],digest)
            self.assertEqual(metadata['display']['layout'],'1x5')
            from tools.check_doris_phy_semantics_7000 import vegetation_confusion_map, SEMANTIC_PALETTE
            with Image.open(root/'paper/spatial/phy_semantic_check.png') as image:
                self.assertEqual(image.size,(196,196))
                expected = np.asarray(SEMANTIC_PALETTE,np.uint8)[vegetation_confusion_map(self.paper_fixture())[30:226,30:226]]
                np.testing.assert_array_equal(np.array(image),expected)
            semantic = json.loads((root/'paper/semantic_statistics.json').read_text(encoding='utf-8'))
            self.assertEqual(semantic['groups'],metadata['display']['semantic_check']['groups'])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            self.assertEqual(original.read_bytes(),b'preserved original figure fixture')
            self.assertEqual(render_phy_paper(source),files)

    def test_paper_refuses_silent_clipping_and_nonrectangular_valid_regions(self):
        from tools.build_doris_mechanism_figures import phy_paper_plate
        maps = self.paper_fixture()
        with self.assertRaisesRegex(ValueError,'excludes saved values'):
            phy_paper_plate(maps,delta_range=(-.01,0))
        maps['valid_mask'][100,100] = False
        with self.assertRaisesRegex(ValueError,'rectangular valid region'):
            phy_paper_plate(maps)

    def test_four_panel_overview_clean_rgb_and_native_color_references(self):
        from tools.build_doris_mechanism_figures import online_plate, render_online_node, plt
        maps,reference = fixture()
        add_reference_evidence(maps,reference,ANNOTATION)
        fig = online_plate([maps],'phys')
        try:
            axes = [axis for axis in fig.axes if axis.images]
            self.assertEqual(len(axes),4)
            self.assertEqual(len(axes[0].patches),0)
            np.testing.assert_array_equal(axes[0].images[0].get_array(),maps['target_image'])
            self.assertIn('Teacher confidence',axes[1].get_title())
            self.assertIn('Pre-gate weight mean',fig._supxlabel.get_text())
            self.assertIn('VI-driven response',fig._supxlabel.get_text())
        finally:plt.close(fig)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            files = render_online_node(maps,directory,7000,render_timing='checkpoint_replay')
            self.assertEqual(len(files),1)
            spatial = Path(directory)/'spatial/iter_007000'
            with Image.open(spatial/'reference_label.png') as image:
                np.testing.assert_array_equal(np.array(image),np.asarray(ANNOTATION['palette'],np.uint8)[reference])
            with Image.open(spatial/'w_conf.png') as image:
                self.assertEqual(image.size,reference.shape[::-1])


if __name__=='__main__':
    unittest.main()
