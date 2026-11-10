"""Branch-level output convergence checks; synthetic data are not experiment results."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import test_minimal_output as helpers
from test_minimal_output import ROOT, torch


@unittest.skipIf(torch is None,'Torch unavailable')
class BranchOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('branch_observation',ROOT/'mmseg/utils/mechanism.py')
        cls.obs=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.obs)

    def model(self):
        from test_doris_mechanism import MechanismTests
        return MechanismTests().model()

    def collector(self,folder,model):
        obj=object.__new__(self.obs.MechanismCollector)
        obj.root=Path(folder);obj.pending=[];obj.window_start=0;obj.model=model
        obj.cfg=dict(output_mode='branches',sample_interval=100,summary_interval=500,
            probe_interval=500,probe_enabled=False,save_raw=True,make_figures=True,detailed_output=False)
        obj.manifest=dict(experiment_name='synthetic_ablation')
        obj.runner=SimpleNamespace(iter=0,logger=SimpleNamespace(info=lambda *a:None))
        return obj

    def record(self,model):
        label=(torch.arange(64).reshape(1,8,8)%5).long()
        valid=torch.ones(1,1,8,8).bool();valid[:,:,:1]=False
        base=torch.ones_like(label).float()*.8
        phy=base*.9 if model.enable_phy else base
        ps=phy*.9 if model.enable_struct else phy
        final=ps*.95 if model.enable_proto else ps
        if model.enable_proto:
            model.proto_gate.collect_diagnostics=True
            model.proto_gate.update_target(torch.randn(1,4,8,8),label,torch.ones_like(base),ps,iter=model.local_iter)
        struct={'R_struct':base} if model.enable_struct else None
        edge=base[:,None] if model.enable_struct_edge else None
        return self.obs.training_record(model,torch.ones_like(base),label,ps,final,valid,struct,
            edge,edge*.002 if edge is not None else None,w_base=base,w_t_phy=phy,r_phy=base[:,None])

    def test_all_seven_masks_500_1000_core_csv_and_resume(self):
        from tools.check_doris_stage1_configs import VARIANTS
        for suffix,flags in VARIANTS[1:]:
            with self.subTest(suffix=suffix),tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
                model=self.model()
                for name,value in zip(('phy','struct','proto','struct_edge'),flags):setattr(model,'enable_'+name,value)
                if flags[0]:model.phy_warmup_iters=model.phy_rampup_iters=0
                if flags[1]:model.struct_gate.struct_warmup_iters=model.struct_gate.struct_rampup_iters=0
                if flags[2]:model.proto_gate.target_update_start=model.proto_gate.weight_start=0
                obj=self.collector(folder,model)
                for step in range(100,1001,100):
                    model.local_iter=step-1;model.mechanism_records.append(self.record(model))
                    obj.runner.iter=step-1;obj.consume()
                path=obj.root/'metrics/branch_mechanism_500.csv'
                with path.open() as stream:rows=list(csv.DictReader(stream))
                self.assertEqual([r['iter'] for r in rows],['500','1000'])
                self.assertEqual(rows[0]['n_sampled_batches'],'5')
                for name,enabled in zip(('phy','struct','proto','edge'),flags):
                    for field in self.obs.BRANCH_FIELDS[name]:self.assertEqual(rows[0][field]!='NA',enabled)
                self.assertFalse((obj.root/'metrics/mechanism_500.csv').exists())
                self.assertFalse((obj.root/'metrics/proto_selection_500.csv').exists())
                self.assertFalse(any('e_img' in key or 'loss_phy_pos' in key for key in rows[0]))
                self.obs.append_unique_csv(path,rows,['iter'])
                with path.open() as stream:self.assertEqual(len(list(csv.DictReader(stream))),2)

    def test_warmup_is_na_and_struct_input_is_after_physiology(self):
        model=self.model();model.local_iter=500
        obj=self.collector(tempfile.mkdtemp(prefix='branch_na_',dir=ROOT/'outputs'),model)
        record=self.record(model);obj.pending=[record];obj.summarize(500)
        with (obj.root/'metrics/branch_mechanism_500.csv').open() as stream:row=next(csv.DictReader(stream))
        self.assertEqual(row['phy_weight_after_mean'],'NA')
        self.assertEqual(row['proto_target_update_ratio'],'NA')
        self.assertEqual(row['phy_state'],'warmup')
        before=record['branch']['struct_weight_before_mean']
        after=record['branch']['struct_weight_after_mean']
        self.assertAlmostEqual(before[0]/before[1],.8*.9,places=6)
        self.assertAlmostEqual(after[0]/after[1],.8*.9*.9,places=6)

    def test_unified_probe_only_needed_arrays_no_online_png_and_state_safe(self):
        model=self.model()
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            obj=self.collector(folder,model);obj.identity='fixture'
            image=torch.randn(1,6,16,16)
            obj.sample=(image,torch.rand(1,3,16,16),[dict(filename='fixed.tif',img_norm_cfg=dict(mean=[0]*6,std=[1]*6))])
            obj.probe_cfg=dict(margins=[1,1,1,1])
            (obj.root/'fixed_probe').mkdir()
            (obj.root/'fixed_probe/sample_manifest.json').write_text(json.dumps(dict(tensor_sha256='fixed')))
            state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
            for step in (500,1000):model.local_iter=step;obj.snapshot(step)
            paths=sorted((obj.root/'fixed_probe/selected_arrays').glob('*.npz'))
            self.assertEqual(len(paths),2);self.assertFalse(list(obj.root.rglob('*.png')))
            with np.load(paths[0]) as a,np.load(paths[1]) as b:
                self.assertIn('w_before_struct',a);self.assertIn('w_before_proto',a)
                self.assertIn('teacher_pseudo_boundary',a);self.assertIn('edge_supervision',a)
                self.assertNotIn('e_prob',a);self.assertNotIn('e_img',a);self.assertNotIn('r_pair',a)
                np.testing.assert_array_equal(a['target_image'],b['target_image'])
                self.assertTrue((a['edge_supervision'][~a['teacher_pseudo_boundary']]==0).all())
            for key,value in model.state_dict().items():torch.testing.assert_close(value,state[key],rtol=0,atol=0)
            torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
            obj.snapshot(1000);self.assertEqual(len(list(paths[0].parent.glob('*.npz'))),2)


@unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Matplotlib unavailable')
class BranchFigureTests(unittest.TestCase):
    def test_forty_nodes_four_plates_and_single_branch_no_extra_outputs(self):
        spec=importlib.util.spec_from_file_location('core_branch_figures',ROOT/'tools/build_doris_mechanism_figures.py')
        plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)
        folder=Path(tempfile.mkdtemp(prefix='branch_figure_synthetic_',dir=ROOT/'outputs'))
        arrays=folder/'fixed_probe/selected_arrays';arrays.mkdir(parents=True)
        (folder/'manifest.json').write_text(json.dumps(dict(output_mode='branches',config_sha256='fixture')))
        source=folder/'physiology/data/phy_vi_statistics.csv';source.parent.mkdir(parents=True)
        with source.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['image_group','index','class_id','class_name','count','median','Q1','Q3'])
            writer.writeheader()
            for index in ('ndvi','ndre','cire'):
                for cls,name in enumerate(('Goji','Tree','Bare soil','Built-up','Road')):
                    writer.writerow(dict(image_group='ALL_BALANCED_GROUPS',index=index,class_id=cls,
                        class_name=name,count=10,median=.2+.1*cls,Q1=.1+.1*cls,Q3=.3+.1*cls))
        for step in range(500,20001,500):
            x=np.tile(np.linspace(0,1,16,dtype=np.float32),(16,1));valid=np.ones((16,16),bool)
            maps={key:x for key in ('r_conf','r_loc','r_adj','r_prob','r_struct','r_phy',
                'w_before_phy','w_after_phy','w_before_proto','w_after_proto','r_edge')}
            maps.update(target_image=np.dstack([x]*3),valid_mask=valid,m_phy=x>.5,
                adjacency_available=valid,semantic_available=valid,delta_w_phy=-x*.1,delta_w_proto=-x*.02,
                proto_active=np.array(step>=14000),teacher_pseudo_boundary=x>.7,edge_supervision=np.where(x>.7,x*.002,0))
            maps['metadata']=np.array(json.dumps(dict(iter=step,config_sha256='fixture',
                sample_id='SYNTHETIC_FIXED_SAMPLE',sample_tensor_sha256='fixed',synthetic_fixture=True,
                data_source='fixed_probe_recomputed',edge_supervision_max=.002,
                modules=dict(phy=True,struct=True,proto=True,edge=True),
                states=dict(phy='warmup' if step<=6000 else 'active',struct='active',
                            proto='warmup' if step<14000 else 'active',edge='active'))))
            np.savez_compressed(arrays/('iter_%06d.npz'%step),**maps)
        output=plot.build_figures(folder)
        self.assertEqual(len(list(output.glob('*.pdf'))),4);self.assertEqual(len(list(output.glob('*.png'))),4)
        self.assertFalse(list(folder.glob('fixed_probe/snapshots/*.png')))
        selection=json.loads((output/'branch_figure_selection.json').read_text())
        self.assertIn(13500,selection['selected_iterations']['proto']);self.assertIn(14000,selection['selected_iterations']['proto'])
        for p in output.glob('*.png'):
            with Image.open(p) as img:self.assertGreaterEqual(img.info['dpi'][0],299)
        stamps={p.name:p.stat().st_mtime_ns for p in output.glob('*.png')}
        plot.build_figures(folder)
        self.assertEqual(stamps,{p.name:p.stat().st_mtime_ns for p in output.glob('*.png')})
        single=folder/'proto_only';single_arrays=single/'fixed_probe/selected_arrays';single_arrays.mkdir(parents=True)
        (single/'manifest.json').write_text(json.dumps(dict(output_mode='branches',config_sha256='fixture')))
        for p in arrays.glob('*.npz'):
            with np.load(p) as saved:maps={k:saved[k] for k in saved.files}
            meta=json.loads(str(maps['metadata']));meta['modules']=dict(phy=False,struct=False,proto=True,edge=False)
            maps['metadata']=np.array(json.dumps(meta));np.savez_compressed(single_arrays/p.name,**maps)
        separate=plot.build_figures(single)
        self.assertEqual(len(list(separate.glob('*.pdf'))),1)
        self.assertEqual(len(list(separate.glob('*.png'))),1)
        self.assertTrue((separate/'fig_proto_branch.png').exists())
        self.assertEqual(plot.plt.get_fignums(),[])
        print('SYNTHETIC core branch QA:',output)


if __name__=='__main__':unittest.main()
