"""Synthetic data validate the two physiology candidates; no experiment claims."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class PhysiologyFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('phy_figures',
            ROOT / 'tools/build_doris_physiology_figures.py')
        cls.plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.plot)

    def test_compact_full_does_not_render_a_third_physiology_plate(self):
        from tools.build_doris_mechanism_figures import render_node
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            values=np.full((8,8),.7,np.float32)
            maps={key:values for key in ('r_conf','r_loc','r_adj','r_prob','r_struct',
                'w_conf','w_base','w_t_ps','w_pre_proto','w_t_proto','delta_w_proto',
                'e_prob','e_img','r_pair','r_edge','boundary_weight')}
            maps.update(target_image=np.dstack([values]*3),valid_mask=np.ones((8,8),bool),
                adjacency_available=np.ones((8,8),bool),semantic_available=np.ones((8,8),bool),
                proto_active=np.array(True),metadata=np.array(json.dumps(dict(iter=10000,
                    sample_id='SYNTHETIC_FULL',synthetic_fixture=True,physiology_compact=True,
                    modules=dict(phy=True,struct=True,proto=True,edge=True)))))
            render_node(maps,folder,10000)
            self.assertEqual(len(list(Path(folder).glob('*.png'))),3)
            self.assertFalse(list(Path(folder).glob('*physiology*')))

    def test_source_sampling_empty_classes_normalization_joint_score_and_cache(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            root = Path(folder)
            for name in ('rgb', 'ann', 'ndvi', 'ndre', 'cire'):
                (root / (name + '_dir/train')).mkdir(parents=True)
            rng = np.random.default_rng(1)
            for name in ('sceneA_0_16_0_16.tif', 'sceneA_16_32_0_16.tif', 'sceneB_0_16_0_16.tif'):
                labels = (np.arange(256).reshape(16,16) % 4 + 1).astype(np.uint8)  # Road absent.
                labels[0,0] = 255
                Image.fromarray(labels).save(root / 'ann_dir/train' / name)
                Image.fromarray(np.zeros((16,16,3),np.uint8)).save(root / 'rgb_dir/train' / name)
                for index in ('ndvi','ndre','cire'):
                    values = labels.astype(np.float32)*.1 + rng.normal(0,.01,(16,16)).astype(np.float32)
                    values[0,0] = np.nan
                    Image.fromarray(values).save(root / (index + '_dir/train') / name)
            source = dict(type='GojiDataset', data_root=str(root), img_dir='rgb_dir/train',
                ann_dir='ann_dir/train', use_vi=True, vi_names=['ndvi','ndre','cire'],
                synthetic_fixture=True, vi_norm_cfg=dict(ndvi=dict(min=-1,max=1),
                    ndre=dict(min=-1,max=1), cire=dict(min=0,max=5)))
            output = root / 'analysis/physiology'
            self.plot.analyze_vi(source, output, per_tile_class=8, per_group_class=7)
            with (output / 'data/phy_vi_statistics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            pooled = {(r['class_id'], r['index']): r for r in rows if r['image_group']=='ALL_BALANCED_GROUPS'}
            self.assertEqual(pooled[('4','ndvi')]['count'], '0')
            self.assertEqual(pooled[('4','ndvi')]['mean'], '')
            self.assertEqual(pooled[('0','ndvi')]['count'], '14')
            with (output / 'data/phy_vi_separability.csv').open() as stream:
                scores = list(csv.DictReader(stream))
            self.assertEqual(len(scores), 7)
            a,b=pooled[('0','ndvi')],pooled[('1','ndvi')]
            expected=(float(a['mean'])-float(b['mean']))**2/(float(a['std'])**2+float(b['std'])**2)
            self.assertAlmostEqual(float(scores[0]['fisher_trace_ratio']), expected, places=4)
            normalized = self.plot.normalize_goji_index(np.array([-2.,0.,2.],np.float32),'ndvi',source['vi_norm_cfg'])
            np.testing.assert_array_equal(normalized, [0.,.5,1.])
            self.assertEqual(len(list(output.rglob('*.*'))), 4)
            with patch.object(self.plot, 'prepare_goji_index', side_effect=AssertionError('dataset reread')):
                self.plot.analyze_vi(source, output, per_tile_class=8, per_group_class=7)
            data_only=root/'data_only'
            self.plot.analyze_vi(source,data_only,per_tile_class=8,per_group_class=7,make_figure=False)
            self.assertEqual(len(list(data_only.rglob('*.csv'))),2)
            self.assertFalse(list(data_only.rglob('*.png')))
            self.assertFalse(list(data_only.rglob('*.pdf')))
            with self.assertRaises(ValueError):
                self.plot.analyze_vi(dict(source, ann_dir='ann_dir/test'), output)

    def test_response_forty_nodes_only_one_evolution_plate_and_repeat_safe(self):
        folder = Path(tempfile.mkdtemp(prefix='phy_evidence_synthetic_',dir=ROOT/'outputs'))
        arrays = folder/'physiology/arrays'; arrays.mkdir(parents=True)
        (folder/'manifest.json').write_text(json.dumps(dict(phy_warmup_start=6000,phy_rampup_iters=4000)))
        shape = (16,16); valid=np.ones(shape,bool); valid[:1]=False
        m_phy=np.zeros(shape,bool);m_phy[:,:8]=True
        for step in range(500,20001,500):
            r=np.tile(np.linspace(0,1,16,dtype=np.float32),(16,1))
            before=np.full(shape,.8,np.float32)*valid
            rho=np.clip((step-6000)/4000,0,1)
            gate=np.where(r<.30,(.82+.18*r)*.5,.82+.18*r)
            after=before*np.where(m_phy,1-rho+rho*gate,1)
            meta=dict(iter=step,sample_id='SYNTHETIC_FIXED_SAMPLE',sample_tensor_sha256='fixed',
                config_sha256='synthetic',data_source='fixed_probe_recomputed',synthetic_fixture=True,
                states=dict(phy='warmup' if step<=6000 else 'active'))
            np.savez_compressed(arrays/('iter_%06d_phy.npz'%step),r_phy=r,m_phy=m_phy,
                w_before_phy=before,w_t_phy=after,delta_w_phy=after-before,
                valid_mask=valid,target_image=np.dstack([r,r,r]),metadata=np.array(json.dumps(meta)))
        output=self.plot.build_phy_response_figure(folder)
        self.assertEqual(len(list(output.glob('*.png'))),1)
        self.assertEqual(len(list(output.glob('*.pdf'))),1)
        with Image.open(output/'fig_phy_response_evolution.png') as image:
            stamp=json.loads(image.info['Description'])
            self.assertEqual(stamp['selected'],[2000,6000,10000,15000,20000])
            self.assertGreaterEqual(image.info['dpi'][0],299)
        stamps={p.name:p.stat().st_mtime_ns for p in output.iterdir()}
        self.plot.build_phy_response_figure(folder)
        self.assertEqual(stamps,{p.name:p.stat().st_mtime_ns for p in output.iterdir()})
        self.assertEqual(self.plot.plt.get_fignums(),[])
        print('SYNTHETIC PHY-2 QA:',output)


if __name__ == '__main__':
    unittest.main()
