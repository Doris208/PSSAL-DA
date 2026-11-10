"""CPU tests using MMCV 1.4 source methods, tiny models and real .pth files.

The source harness replaces dependency imports/FileClient, not the CheckpointHook
or EvalHook algorithms. It does not claim a full MMCV/CUDA training acceptance.
"""
import ast
import copy
import importlib.util
import json
import os.path as osp
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np
from test_minimal_output import ROOT, static_config, torch
from tools.doris_checkpoint_plan import (applies, checkpoint_plan,
    configure_checkpoint_policy, check_output_directory)

MMCV = ROOT/'outputs/no3vi_check_deps/mmcv'


def configs():
    return sorted((ROOT/'configs/hrda/doris').glob('*/*.py'))


class PlanTests(unittest.TestCase):
    def test_all_24_configs_preserve_algorithm_evaluation_and_workdirs(self):
        self.assertEqual(len(configs()),24)
        directories = set()
        for path in configs():
            cfg = static_config(path.relative_to(ROOT).as_posix())
            original = copy.deepcopy(cfg)
            self.assertTrue(configure_checkpoint_policy(cfg))
            self.assertEqual(cfg['checkpoint_config']['interval'],2000)
            self.assertEqual(cfg['checkpoint_config']['max_keep_ckpts'],-1)
            self.assertTrue(cfg['checkpoint_config']['save_optimizer'])
            self.assertTrue(cfg['evaluation'].pop('save_best_goji'))
            cfg.pop('checkpoint_config'); original.pop('checkpoint_config')
            self.assertEqual(cfg,original)
            directories.add(cfg['work_dir'])
        self.assertEqual(len(directories),24)

    def test_actual_direction_specific_plans_and_no_fictitious_pair_warmup(self):
        for direction, update, weight in [('goji26RGB2goji30RGB',3000,6000),
                ('goji26IRRG2goji30RGB',4000,8000),('goji30IRRG2goji26IRRG',3000,6000)]:
            for path in (ROOT/'configs/hrda/doris'/direction).glob('*.py'):
                cfg=static_config(path.relative_to(ROOT).as_posix()); plan=checkpoint_plan(cfg)
                self.assertEqual(plan['periodic'],list(range(2000,20001,2000)))
                self.assertEqual(len(plan['all_iterations']),len(set(plan['all_iterations'])))
                self.assertFalse(any(e['branch']=='edge_pair' for e in plan['events']))
                if cfg['uda']['proto_cfg']['enable']:
                    self.assertTrue(any(e['kind']=='target_update_eligible' and e['local_iter']==update
                                        for e in plan['events']))
                    self.assertTrue(any(e['kind']=='weight_eligible' and e['local_iter']==weight
                                        for e in plan['events']))
                if path.stem.endswith('_proto_struct_edge'):
                    expected = [3000,3500,5000,5500,6500,10500] if update==3000 else \
                               [4500,5000,5500,6500,8500,10500]
                    self.assertEqual(plan['extra'],expected)
                if path.stem.endswith('_3vi'):
                    self.assertEqual(plan['extra'],[])

    def test_occupied_directory_requires_explicit_resume_and_other_jobs_unchanged(self):
        cfg=static_config(configs()[0].relative_to(ROOT).as_posix())
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            cfg['work_dir']=folder
            check_output_directory(cfg)
            (Path(folder)/'old.log').write_text('keep')
            with self.assertRaises(FileExistsError): check_output_directory(cfg)
            cfg['resume_from']='checkpoint.pth';check_output_directory(cfg)
        other=dict(name='doris_goji26RGB2goji30RGB_daformer_3vi')
        before=copy.deepcopy(other)
        self.assertFalse(configure_checkpoint_policy(other));self.assertEqual(other,before)


def source_classes(path, scope):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope


@unittest.skipIf(torch is None or not MMCV.is_dir(),'Torch or MMCV 1.4 source unavailable')
class SavingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Torch's lazy optimizer imports before scoped module patches;
        # removing those imports on cleanup would duplicate C++ registrations.
        torch.optim.Adam(torch.nn.Linear(1,1).parameters())
        class FileClient:
            name='local';allow_symlink=True
            @classmethod
            def infer_client(cls,*args):return cls()
            join_path=staticmethod(osp.join)
            isfile=staticmethod(osp.isfile)
            remove=staticmethod(lambda p:Path(p).unlink())
        registry=SimpleNamespace(register_module=lambda:lambda cls:cls)
        scope=dict(HOOKS=registry,osp=osp,warnings=warnings,FileClient=FileClient,
            master_only=lambda f:f,allreduce_params=lambda x:None,inf=float('inf'),
            is_seq_of=lambda x,t:all(isinstance(v,t) for v in x),
            DataLoader=torch.utils.data.DataLoader)
        source_classes(MMCV/'runner/hooks/hook.py',scope)
        source_classes(MMCV/'runner/hooks/checkpoint.py',scope)
        source_classes(MMCV/'runner/hooks/evaluation.py',scope)
        cls.original=SimpleNamespace(**scope)
        modules={'mmcv.runner':SimpleNamespace(HOOKS=registry,CheckpointHook=scope['CheckpointHook']),
            'mmcv.runner.dist_utils':SimpleNamespace(master_only=lambda f:f)}
        cls.imports=patch.dict(sys.modules,modules);cls.imports.start();cls.addClassCleanup(cls.imports.stop)

        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,ROOT/path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            return module
        cls.policy=load('policy_under_test','mmseg/utils/checkpoint_policy.py')
        cls.policy_import=patch.dict(sys.modules,{'mmseg.utils.checkpoint_policy':cls.policy})
        cls.policy_import.start();cls.addClassCleanup(cls.policy_import.stop)
        best=load('best_under_test','mmseg/core/evaluation/goji_best.py')
        scope.update(_EvalHook=scope['EvalHook'],_DistEvalHook=scope['DistEvalHook'],
                     GojiBestMixin=best.GojiBestMixin)
        source_classes(ROOT/'mmseg/core/evaluation/eval_hooks.py',scope)
        cls.Eval=scope['EvalHook'];cls.DistEval=scope['DistEvalHook']

        def save(model,filename,optimizer=None,meta=None):
            torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                            meta=copy.deepcopy(meta)),filename)
        runner_scope=dict(osp=osp,platform=platform,shutil=shutil,save_checkpoint=save,
            mmcv=SimpleNamespace(symlink=Mock()))
        tree=ast.parse((MMCV/'runner/iter_based_runner.py').read_text())
        runner_node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IterBasedRunner')
        method=next(n for n in runner_node.body if isinstance(n,ast.FunctionDef) and n.name=='save_checkpoint')
        method.name='runner_save_checkpoint' # Keep its global serializer binding separate.
        exec(compile(ast.Module(body=[method],type_ignores=[]),'mmcv_runner_save','exec'),runner_scope)
        cls.runner_class=type('Runner',(),dict(save_checkpoint=runner_scope['runner_save_checkpoint']))
        cls.symlink=runner_scope['mmcv'].symlink

    def runner(self,folder,model=None):
        runner=self.runner_class();runner.model=model if model is not None else torch.nn.Linear(2,2)
        runner.optimizer=torch.optim.Adam(runner.model.parameters())
        runner.work_dir=folder;runner.iter=0;runner.epoch=0;runner._max_iters=20000;runner.max_iters=20000
        runner.rank=0;runner.meta={};runner.logger=Mock();runner.hooks=[]
        runner.log_buffer=SimpleNamespace(output={},ready=False)
        return runner

    def evaluation(self,runner,distributed=False):
        class Dataset(torch.utils.data.Dataset):
            CLASSES=('goji','vegetation','building','water','road')
            def __len__(self):return 1
            def __getitem__(self,index):return 0
            def evaluate(self,results,**kwargs):return results
        loader=torch.utils.data.DataLoader(Dataset())
        options=dict(interval=1000,save_best='mIoU',save_best_goji=True,test_fn=lambda *a:None)
        if distributed:options['broadcast_bn_buffer']=False
        hook=(self.DistEval if distributed else self.Eval)(loader,**options)
        hook.before_run(runner)
        return hook

    def validate(self,runner,hook,step,miou,goji):
        runner.iter=step-1
        score=hook.evaluate(runner,dict(mIoU=miou,**{'IoU.goji':goji,'Veg.mIoU':1.0}))
        hook._save_ckpt(runner,score)

    def test_union_keeps_every_node_and_latest_with_no_duplicate_writes(self):
        cfg=static_config('configs/hrda/doris/goji26RGB2goji30RGB/'
            'doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py')
        plan=checkpoint_plan(cfg)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            runner=self.runner(folder)
            hook=self.policy.DorisCheckpointHook(interval=2000,by_epoch=False,
                key_iterations=plan['extra'],max_keep_ckpts=-1)
            runner.hooks=[hook];hook.before_run(runner)
            with patch.object(runner,'save_checkpoint',wraps=runner.save_checkpoint) as save:
                for step in range(1,20001):
                    runner.iter=step-1;hook.after_train_iter(runner)
                self.assertEqual(save.call_count,len(plan['all_iterations']))
            files=sorted(int(p.stem[5:]) for p in Path(folder).glob('iter_*.pth'))
            self.assertEqual(files,plan['all_iterations'])
            latest=torch.load(Path(folder)/'latest.pth')
            self.assertEqual(latest['meta']['iter'],19999)
            self.assertEqual(latest['meta']['doris_checkpoint']['completed_iter'],20000)
            with self.assertRaises(FileExistsError):hook.after_train_iter(runner)

    def test_independent_bests_ties_nan_zero_and_resume_from_either_best(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            runner=self.runner(folder);hook=self.evaluation(runner)
            self.validate(runner,hook,1000,0.,np.float32(0.))
            self.validate(runner,hook,2000,.7,.8)
            both=torch.load(Path(folder)/'best_mIoU_iter_2000.pth')['meta']['hook_msgs']
            self.assertEqual(both['best_goji_score'],.8)
            self.validate(runner,hook,3000,.6,.9) # Goji improves alone.
            self.validate(runner,hook,4000,.75,.85) # mIoU improves alone.
            self.validate(runner,hook,5000,.75,.9) # ties do not save.
            self.validate(runner,hook,6000,float('nan'),float('nan'))
            self.assertEqual({p.name for p in Path(folder).glob('*.pth')},
                {'best_mIoU_iter_4000.pth','best_goji_mIoU_iter_3000.pth'})
            report=json.loads((Path(folder)/'checkpoint_best.json').read_text())
            self.assertFalse(report['same_iter'])
            self.assertEqual(report['IoU.goji']['score'],.9)
            for filename in ('best_mIoU_iter_4000.pth','best_goji_mIoU_iter_3000.pth'):
                state=torch.load(Path(folder)/filename)
                resumed=self.runner(folder);resumed.meta=state['meta'];resumed.iter=state['meta']['doris_checkpoint']['completed_iter']
                loaded=self.evaluation(resumed)
                self.validate(resumed,loaded,7000,.5,.7)
                self.assertEqual(resumed.meta['hook_msgs']['best_goji_score'],.9)
                self.assertFalse((Path(folder)/'best_goji_mIoU_iter_7000.pth').exists())

    def test_periodic_after_validation_contains_both_best_records(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            runner=self.runner(folder);evaluation=self.evaluation(runner)
            hook=self.policy.DorisCheckpointHook(interval=2000,by_epoch=False)
            runner.hooks=[hook];hook.before_run(runner)
            self.validate(runner,evaluation,2000,.6,.8);hook.after_train_iter(runner)
            meta=torch.load(Path(folder)/'latest.pth')['meta']['hook_msgs']
            self.assertEqual(meta['best_score'],.6);self.assertEqual(meta['best_goji_score'],.8)
            self.assertEqual(meta['best_miou_iter'],2000);self.assertEqual(meta['best_goji_iter'],2000)

    def test_preexisting_bests_are_preserved_when_new_bests_appear(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            runner=self.runner(folder);hook=self.evaluation(runner)
            self.validate(runner,hook,1000,.5,.6)
            old={p:p.read_bytes() for p in Path(folder).glob('*.pth')}
            resumed=self.runner(folder);resumed.meta=copy.deepcopy(runner.meta);resumed.iter=1000
            hook=self.evaluation(resumed)
            self.validate(resumed,hook,2000,.7,.9)
            for p,content in old.items():self.assertEqual(p.read_bytes(),content)

    def test_future_checkpoint_and_low_space_fail_without_deletion(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            runner=self.runner(folder)
            future=Path(folder)/'best_mIoU_iter_5000.pth';future.write_bytes(b'history')
            hook=self.policy.DorisCheckpointHook(interval=2000,by_epoch=False);runner.hooks=[hook]
            with self.assertRaises(FileExistsError):hook.before_run(runner)
            with patch.object(self.policy.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError,'no history was deleted'):
                    self.policy.prepare_save(runner,Path(folder)/'iter_2000.pth')
            self.assertEqual(future.read_bytes(),b'history')

    def test_write_failure_preserves_previous_best_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            runner=self.runner(folder);hook=self.evaluation(runner)
            self.validate(runner,hook,1000,.5,.6)
            before={p:p.read_bytes() for p in Path(folder).glob('*.pth')}
            with patch.object(runner,'save_checkpoint',side_effect=OSError('quota exhausted')):
                with self.assertRaisesRegex(OSError,'quota'):
                    self.validate(runner,hook,2000,.7,.8)
            for path,content in before.items():self.assertEqual(path.read_bytes(),content)

    def test_legacy_nan_and_missing_goji_history_can_start_valid_records(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            runner=self.runner(folder);runner.iter=5000
            runner.meta={'hook_msgs':{'best_score':float('nan')}}
            hook=self.evaluation(runner)
            self.validate(runner,hook,6000,.7,.8)
            report=json.loads((Path(folder)/'checkpoint_best.json').read_text())
            self.assertEqual(report['goji_history_start_iter'],5000)
            self.assertEqual(report['mIoU']['score'],.7)
            self.assertTrue(report['same_iter'])

    def test_regular_file_latest_resume_restores_completed_iteration(self):
        import re
        tree=ast.parse((ROOT/'mmseg/apis/train.py').read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_segmentor')
        block=next(n for n in function.body if isinstance(n,ast.If) and
                   ast.dump(n.test)==ast.dump(ast.parse('cfg.resume_from',mode='eval').body))
        class Config(dict):
            __getattr__=dict.__getitem__
        class Runner:
            _iter=1999
            meta={'doris_checkpoint':{'completed_iter':2000}}
            model=SimpleNamespace(local_iter=0)
            @property
            def iter(self):return self._iter
            def resume(self,path):pass
        runner=Runner()
        cfg=Config(resume_from='latest.pth',load_from=None,uda=dict(type='DACS'),runner=dict(type='IterBasedRunner'))
        exec(compile(ast.Module(body=[block],type_ignores=[]),'resume','exec'),
            dict(cfg=cfg,runner=runner,osp=osp,re=re,logger=Mock()))
        self.assertEqual(runner.iter,2000);self.assertEqual(runner.model.local_iter,2000)

    def test_single_and_distributed_hooks_use_one_validation_pass(self):
        for distributed in (False,True):
            with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
                runner=self.runner(folder);hook=self.evaluation(runner,distributed)
                runner.iter=999
                inference=Mock(return_value={'mIoU':.7,'IoU.goji':.8})
                with patch.dict(sys.modules,{'mmseg.apis':SimpleNamespace(
                        single_gpu_test=inference,multi_gpu_test=inference)}):
                    hook._do_evaluate(runner)
                self.assertEqual(inference.call_count,1)
                self.assertEqual(len(list(Path(folder).glob('best_*.pth'))),2)

    def test_full_component_state_roundtrip_including_populated_domain_banks(self):
        from test_doris_mechanism import MechanismTests
        from test_doris_stage1 import new_model
        model=MechanismTests().model()
        model.proto_gate.target_prototypes.copy_(model.proto_gate.source_prototypes*.7)
        model.proto_gate.target_valid_proto.fill_(True)
        model.struct_edge_pair_prior.fill_(.3);model.struct_edge_pair_prior_ready.fill_(True)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            runner=self.runner(folder,model);hook=self.evaluation(runner)
            self.validate(runner,hook,10000,.72,.83)
            saved=torch.load(Path(folder)/'best_goji_mIoU_iter_10000.pth')
            restored=new_model();restored.model=copy.deepcopy(model.model);restored.ema_model=copy.deepcopy(model.ema_model)
            # Fresh ProtoGate has empty dynamic banks; its existing loader allocates them.
            self.assertEqual(restored.proto_gate.prototypes.numel(),0)
            with torch.no_grad():
                for value in restored.state_dict().values():value.zero_()
            restored.load_state_dict(saved['state_dict'],strict=True)
            for name,value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)
            self.assertIn('optimizer',saved)
            self.assertEqual(saved['meta']['doris_checkpoint']['completed_iter'],10000)

    def test_linux_latest_uses_original_relative_symlink_interface(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            runner=self.runner(folder);runner.iter=1999
            with patch.object(platform,'system',return_value='Linux'):
                runner.save_checkpoint(folder)
            self.symlink.assert_called_with('iter_2000.pth',osp.join(folder,'latest.pth'))


if __name__=='__main__':
    unittest.main()
