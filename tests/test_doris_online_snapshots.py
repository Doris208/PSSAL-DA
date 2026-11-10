"""Real file output through lightweight tensors/Agg; synthetic fixtures are not results."""
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from test_minimal_output import ROOT, torch, load_methods
import test_doris_branch_outputs as branch_tests


LAYOUTS = [('phys',(1,0,0,0)),('struct',(0,1,0,0)),('mparc',(1,1,0,0)),
           ('proto',(0,0,1,0)),('edge',(0,0,0,1)),('pbcr',(0,0,1,1)),('full',(1,1,1,1))]


@unittest.skipIf(torch is None,'Torch unavailable')
class OnlineCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        branch_tests.BranchOutputTests.setUpClass()
        cls.helper=branch_tests.BranchOutputTests()
        cls.obs=branch_tests.BranchOutputTests.obs

    def collector(self,folder,flags):
        model=self.helper.model()
        for key,flag in zip(('phy','struct','proto','struct_edge'),flags):setattr(model,'enable_'+key,bool(flag))
        obj=self.helper.collector(folder,model);obj.identity='synthetic_fixture'
        obj.cfg.update(probe_enabled=True,online_plot=True,strict=True)
        obj.runner.logger=Mock()
        obj.sample=(torch.randn(1,6,16,16),torch.rand(1,3,16,16),
            [dict(filename='SYNTHETIC_FIXED_SAMPLE.tif',img_norm_cfg=dict(mean=[0]*6,std=[255]*6))])
        obj.probe_cfg=dict(margins=[1,1,1,1],channel_order=['R','G','B'])
        (obj.root/'fixed_probe').mkdir(parents=True)
        (obj.root/'fixed_probe/sample_manifest.json').write_text(json.dumps(dict(tensor_sha256='synthetic_fixed')))
        return obj

    def bridge(self,maps,output,completed,**kwargs):
        # Some local installations split Torch and Matplotlib between existing environments.
        # Production uses a direct function call; only this test bridges runtimes.
        source=Path(output).parent/'selected_arrays'/('iter_%06d.npz'%completed)
        code=('import sys,hashlib;from tools.build_doris_mechanism_figures import read_node,render_online_node;'
              'from pathlib import Path;p=Path(sys.argv[1]);'
              'render_online_node(read_node(p),sys.argv[2],int(sys.argv[3]),'
              'array_sha256=hashlib.sha256(p.read_bytes()).hexdigest())')
        subprocess.run([os.environ.get('DORIS_PLOT_PYTHON',sys.executable),'-c',code,
                        str(source),str(output),str(completed)],cwd=ROOT,check=True,capture_output=True,text=True)
        random.random()  # The outer probe guard also covers rendering RNG use.
        return sorted(Path(output).glob('iter_%06d_*.png'%completed))

    def test_seven_live_layouts_500_1000_files_state_and_resume(self):
        if not os.environ.get('DORIS_PLOT_PYTHON') and not importlib.util.find_spec('matplotlib'):
            self.skipTest('Set DORIS_PLOT_PYTHON to an existing Matplotlib interpreter')
        root=Path(tempfile.mkdtemp(prefix='doris_online_synthetic_',dir=ROOT/'outputs'))
        for kind,flags in LAYOUTS:
            with self.subTest(kind=kind):
                obj=self.collector(root/kind,flags);model=obj.model
                model.local_iter=0
                optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
                sum(p.sum() for p in model.parameters()).backward();optimizer.step()
                state=copy.deepcopy(model.state_dict());optim=copy.deepcopy(optimizer.state_dict())
                gradients=[p.grad.clone() for p in model.parameters()]
                modes=[m.training for m in model.modules()]
                with patch.dict(sys.modules,{'tools.build_doris_mechanism_figures':SimpleNamespace(render_online_node=self.bridge)}):
                    for step in range(100,1001,100):
                        model.local_iter=step-1
                        model.mechanism_records.append(self.helper.record(model))
                        model.local_iter=step;obj.runner.iter=step-1
                        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
                        obj.consume()
                        self.assertEqual(rng[0],random.getstate());np.testing.assert_equal(rng[1],np.random.get_state())
                        torch.testing.assert_close(rng[2],torch.get_rng_state(),rtol=0,atol=0)
                    with (obj.root/'metrics/branch_mechanism_500.csv').open() as stream:
                        self.assertEqual([r['iter'] for r in csv.DictReader(stream)],['500','1000'])
                    pngs=sorted((obj.root/'fixed_probe/snapshots').glob('*.png'))
                    self.assertEqual(len(pngs),4 if kind=='full' else 2)
                    self.assertEqual(len(list(obj.root.rglob('*.npz'))),2)
                    self.assertFalse(list(obj.root.rglob('*.pdf')))
                    manifest=json.loads((obj.root/'manifest.json').read_text())
                    for step in (500,1000):
                        entry=manifest['snapshots'][str(step)]
                        self.assertEqual(entry['snapshot_status'],'completed')
                        raw=obj.root/entry['array_path'];digest=hashlib.sha256(raw.read_bytes()).hexdigest()
                        with np.load(raw) as saved:
                            meta=json.loads(str(saved['metadata']))
                            self.assertEqual(meta['iter'],step)
                            if flags[2]:self.assertIn('tau_proto',meta['prototype'])
                        for file in entry['figures']:
                            with Image.open(obj.root/file) as image:
                                info=json.loads(image.info['Description']);image.verify()
                                self.assertEqual(info['iter'],step);self.assertEqual(info['array_sha256'],digest)
                                self.assertEqual(info['states'],meta['states'])
                    for key,value in model.state_dict().items():torch.testing.assert_close(value,state[key],rtol=0,atol=0)
                    for p,g in zip(model.parameters(),gradients):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
                    self.assertEqual(modes,[m.training for m in model.modules()])
                    for key,values in optim['state'].items():
                        for field,value in values.items():torch.testing.assert_close(value,optimizer.state_dict()['state'][key][field],rtol=0,atol=0)
                    if kind=='full':
                        stamps={p:p.stat().st_mtime_ns for p in obj.root.rglob('*') if p.is_file()}
                        with patch.object(self.obs,'probe_arrays',side_effect=AssertionError('duplicate inference')):
                            obj.snapshot(1000)
                        self.assertEqual(stamps,{p:p.stat().st_mtime_ns for p in stamps})
                        obj.window_start=1000;model.local_iter=1499
                        model.mechanism_records.append(self.helper.record(model));model.local_iter=1500
                        obj.runner.iter=1499;obj.consume()
                        with (obj.root/'metrics/branch_mechanism_500.csv').open() as s:
                            self.assertEqual([r['iter'] for r in csv.DictReader(s)],['500','1000','1500'])
                        self.assertEqual(len(list((obj.root/'fixed_probe/snapshots').glob('*.png'))),6)
        print('REAL SYNTHETIC online files:',root)

    def test_schedule_and_hook_precedes_validation(self):
        obj=object.__new__(self.obs.MechanismCollector)
        obj.pending=[];obj.model=SimpleNamespace(mechanism_records=[])
        obj.cfg=dict(summary_interval=500,probe_interval=500,probe_enabled=True)
        obj.runner=SimpleNamespace(iter=0);obj.summarize=Mock();obj.snapshot=Mock()
        for completed in (499,500,999,1000,1499,1500):obj.runner.iter=completed-1;obj.consume()
        self.assertEqual([c.args[0] for c in obj.snapshot.call_args_list],[500,1000,1500])
        hook=load_methods('mmseg/utils/precision_logger.py','MechanismSnapshotHook',
                          ['after_train_iter'],dict(time=SimpleNamespace(time=lambda:0)))()
        events=[]
        snapshot=SimpleNamespace(mechanism=SimpleNamespace(consume=lambda:events.append('snapshot')))
        runner=SimpleNamespace(hooks=[hook,snapshot])
        hook.after_train_iter(runner);events.append('validation')
        self.assertEqual(events,['snapshot','validation'])
        tree=ast.parse((ROOT/'mmseg/apis/train.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='register_hook']
        self.assertTrue(any(any(k.arg=='priority' and isinstance(k.value,ast.Constant) and k.value.value=='ABOVE_NORMAL' for k in c.keywords) for c in calls))

    def test_plot_failures_are_recorded_strict_and_no_fake_completion(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            obj=self.collector(folder,(1,1,1,1));obj.model.local_iter=500
            renderer=Mock(side_effect=RuntimeError('synthetic PNG failure'))
            with patch.dict(sys.modules,{'tools.build_doris_mechanism_figures':SimpleNamespace(render_online_node=renderer)}):
                with self.assertRaisesRegex(RuntimeError,'synthetic PNG failure'):obj.snapshot(500)
                entry=json.loads((obj.root/'manifest.json').read_text())['snapshots']['500']
                self.assertEqual(entry['snapshot_status'],'failed');self.assertIn('RuntimeError',entry['traceback'])
                self.assertTrue(obj.runner.logger.error.called)
                obj.cfg['strict']=False;obj.model.local_iter=1000;obj.snapshot(1000)
                self.assertEqual(obj.manifest['snapshots']['1000']['snapshot_status'],'failed')
            obj.cfg['strict']=True;obj.model.local_iter=500
            empty=SimpleNamespace(render_online_node=lambda *a,**kw:[])
            with patch.dict(sys.modules,{'tools.build_doris_mechanism_figures':empty}):
                with self.assertRaisesRegex(IOError,'required PNGs'):obj.snapshot(500)

    def test_existing_schema4_accepts_output_upgrade_but_rejects_training_changes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            model=self.helper.model()
            model.mechanism_cfg.update(probe_enabled=False,online_plot=False)
            config=dict(model=dict(train_cfg={}),uda=dict(fixed_probe_index=0),
                target_img_norm_cfg=dict(mean=[0]*6,std=[1]*6,to_rgb=False),
                data=dict(train=dict(target=dict(type='GojiDataset',pipeline=[]))),seed=0)
            runner=SimpleNamespace(work_dir=folder,iter=1000,mechanism_config=config,
                                   meta=dict(git_commit='fixture'),logger=Mock())
            old=self.obs.MechanismCollector(runner,model)
            old.record_snapshot(500,snapshot_status='completed',figures=[],array_path='preserved.npz')
            identity=old.identity
            image=torch.zeros(6,16,16);vi=torch.zeros(3,16,16)
            dataset=SimpleNamespace(img_infos=[dict(filename='fixed.tif')])
            class Dataset:
                img_infos=dataset.img_infos
                def __getitem__(self,index):
                    return dict(img=SimpleNamespace(data=image),vi_img=SimpleNamespace(data=vi),
                        img_metas=SimpleNamespace(data=dict(filename='fixed.tif',img_norm_cfg=dict(mean=[0]*6,std=[1]*6))))
            config['uda'].update(mechanism_online_plot=True,mechanism_strict=True)
            model.mechanism_cfg.update(online_plot=True,strict=True,probe_enabled=True)
            with patch.dict(sys.modules,{'mmseg.datasets':SimpleNamespace(build_dataset=lambda cfg:Dataset())}):
                upgraded=self.obs.MechanismCollector(runner,model)
                self.assertEqual(upgraded.identity,identity)
                self.assertEqual(upgraded.root,old.root)
                self.assertEqual(upgraded.manifest['snapshots']['500']['array_path'],'preserved.npz')
                self.assertTrue(upgraded.manifest['snapshot_policy']['online_plot'])
                selected_root=old.root/'resume_runs/existing'
                for relative in ('manifest.json','fixed_probe/sample_manifest.json'):
                    destination=selected_root/relative
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_bytes((old.root/relative).read_bytes())
                (old.root/'active_resume.json').write_text(json.dumps(dict(path='resume_runs/existing')))
                selected=self.obs.MechanismCollector(runner,model)
                self.assertEqual(selected.root,selected_root.resolve())
                config['seed']=99
                with self.assertRaisesRegex(ValueError,'manifest differs'):
                    self.obs.MechanismCollector(runner,model)

    def test_completed_node_survives_conflicting_state_attempt(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            obj=self.collector(folder,(1,1,1,1));obj.cfg['online_plot']=False
            obj.model.local_iter=500;obj.snapshot(500)
            path=obj.root/'fixed_probe/selected_arrays/iter_000500.npz';original=path.read_bytes()
            with torch.no_grad():next(obj.model.parameters()).add_(1)
            with self.assertRaisesRegex(ValueError,'differs in model/sample/config'):obj.snapshot(500)
            self.assertEqual(path.read_bytes(),original)
            entry=obj.manifest['snapshots']['500']
            self.assertEqual(entry['snapshot_status'],'completed')
            self.assertEqual(entry['last_failed_attempt']['snapshot_status'],'failed')

    def test_rollback_preserves_history_and_inherits_only_checkpoint_prefix(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            obj=self.collector(Path(folder)/'analysis',(1,1,1,1))
            obj.base_root=obj.root
            obj.cfg.update(online_plot=False,probe_enabled=False)
            for step in (5000,5500,6000):
                obj.model.local_iter=step-1
                obj.pending=[self.helper.record(obj.model)]
                obj.summarize(step)
            for step in (5000,5500):
                raw=obj.root/'fixed_probe/selected_arrays'/('iter_%06d.npz'%step)
                raw.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(raw,value=np.array([step],np.int64))
                png=obj.root/'fixed_probe/snapshots'/('iter_%06d_mparc.png'%step)
                png.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(2,2)).save(png)
                obj.record_snapshot(step,snapshot_status='completed',array_path=raw.relative_to(obj.root).as_posix(),
                                    figures=[png.relative_to(obj.root).as_posix()])
            original_root=obj.root
            originals={p.relative_to(original_root):p.read_bytes() for p in original_root.rglob('*') if p.is_file()}
            obj.runner.resume_from='latest.pth';obj.runner.iter=5000;obj.window_start=5000
            with patch.object(self.obs.shutil,'copy2',side_effect=OSError('synthetic copy interruption')):
                with self.assertRaisesRegex(OSError,'copy interruption'):obj.prepare_resume(5000)
            self.assertFalse((original_root/'active_resume.json').exists())
            obj.prepare_resume(5000)
            self.assertNotEqual(obj.root,original_root)
            for relative,content in originals.items():self.assertEqual((original_root/relative).read_bytes(),content)
            self.assertEqual(set(obj.manifest['snapshots']),{'5000'})
            self.assertTrue((obj.root/'fixed_probe/selected_arrays/iter_005000.npz').exists())
            self.assertFalse((obj.root/'fixed_probe/selected_arrays/iter_005500.npz').exists())
            self.assertEqual((obj.root/'fixed_probe/selected_arrays/iter_005000.npz').read_bytes(),
                             originals[Path('fixed_probe/selected_arrays/iter_005000.npz')])
            pointer=json.loads((original_root/'active_resume.json').read_text())
            self.assertEqual((original_root/pointer['path']).resolve(),obj.root.resolve())
            for step in range(5100,5501,100):
                obj.model.local_iter=step-1
                obj.model.mechanism_records.append(self.helper.record(obj.model))
                obj.runner.iter=step-1;obj.consume()
            with (obj.root/'metrics/branch_mechanism_500.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([row['iter'] for row in rows],['5000','5500'])
            self.assertEqual(rows[-1]['window_start'],'5000')
            selected=obj.root;obj.prepare_resume(5500)
            self.assertEqual(obj.root,selected)


@unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Matplotlib unavailable')
class OnlineFigureTests(unittest.TestCase):
    def test_summary_entry_resolves_current_resume_segment(self):
        from tools import build_doris_mechanism_figures as plot
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            root=Path(folder)
            selected=root/'resume_runs/active';selected.mkdir(parents=True)
            pointer=root/'active_resume.json'
            pointer.write_text(json.dumps(dict(path='resume_runs/active')))
            self.assertEqual(plot.active_analysis(root),selected.resolve())
            pointer.write_text(json.dumps(dict(path='../outside')))
            with self.assertRaisesRegex(ValueError,'Invalid active resume'):plot.active_analysis(root)

    def test_forty_nodes_arbitrary_selection_two_summaries_and_inactive_panels(self):
        from tools import build_doris_mechanism_figures as plot
        root=Path(tempfile.mkdtemp(prefix='doris_online_summary_synthetic_',dir=ROOT/'outputs'))
        arrays=root/'fixed_probe/selected_arrays';arrays.mkdir(parents=True)
        (root/'manifest.json').write_text(json.dumps(dict(output_mode='branches',snapshot_policy=dict(online_plot=True))))
        x=np.tile(np.linspace(0,1,16,dtype=np.float32),(16,1))
        maps={k:x for k in ('r_phy','r_conf','r_loc','r_adj','r_prob','r_struct','w_before_phy','w_after_phy',
            'w_before_struct','w_after_struct','w_before_proto','w_after_proto','r_edge')}
        maps.update(target_image=np.dstack([x]*3),valid_mask=np.ones((16,16),bool),delta_w_phy=x*0,
                    delta_w_proto=x*0,edge_supervision=x*.006,teacher_pseudo_boundary=x>.6,proto_active=np.array(False))
        for step in range(500,20001,500):
            maps['metadata']=np.array(json.dumps(dict(iter=step,sample_id='SYNTHETIC_SAMPLE',synthetic_fixture=True,
                config_sha256='fixture',sample_tensor_sha256='fixed',data_source='fixed_probe_recomputed',
                modules=dict(phy=True,struct=True,proto=True,edge=True),edge_supervision_max=.006,
                states=dict(phy='warmup',struct='warmup',proto='warmup',edge='ramping'))))
            np.savez_compressed(arrays/('iter_%06d.npz'%step),**maps)
        chosen=[3500,7500,11500,15500,19500]
        output=plot.build_figures(root,iterations=chosen)
        self.assertEqual({p.name for p in output.glob('*.pdf')},{'fig_mparc_summary.pdf','fig_pbcr_summary.pdf'})
        self.assertEqual(len(list(output.glob('*.png'))),2)
        self.assertEqual(json.loads((output/'online_figure_selection.json').read_text())['selected_iterations'],chosen)
        with self.assertRaises(ValueError):plot.build_figures(root,iterations=[3333])
        for kind,flags in [('baseline',(0,0,0,0))]+LAYOUTS:
            meta=json.loads(str(maps['metadata']));meta['modules']=dict(zip(('phy','struct','proto','edge'),map(bool,flags)))
            maps['metadata']=np.array(json.dumps(meta))
            paths=plot.render_online_node(maps,root/'layouts'/kind,20000)
            self.assertEqual(len(paths),0 if kind=='baseline' else 2 if kind=='full' else 1)
            for p in paths:
                with Image.open(p) as image:image.verify()
        fig=plot.online_plate([maps],'pbcr')
        self.assertTrue(any('warmup' in ax.get_ylabel() for ax in fig.axes))
        plot.plt.close(fig)
        self.assertEqual(plot.plt.get_fignums(),[])
        print('SYNTHETIC summary/layout QA:',root)


if __name__=='__main__':unittest.main()
