"""Targeted Baseline replay and aligned-paper checks with synthetic fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from test_minimal_output import ROOT, torch


@unittest.skipIf(torch is None,'Torch unavailable')
class BaselineReplayTests(unittest.TestCase):
    def test_r101_rgb_only_entry_pins_its_checkpoint(self):
        from tools import replay_doris_daformer_r101_no3vi_15000 as entry
        self.assertEqual(str(entry.CHECKPOINT).replace('\\','/'),
            '/root/shared-nvme/work_dirs/doris_no3vi_r101/goji26RGB2goji30RGB_daformer_r101_no3vi/best_mIoU_iter_15000.pth')
        with patch.object(entry,'replay_baseline',return_value=0) as replay:
            self.assertEqual(entry.main([]),0)
        replay.assert_called_once_with([],default_checkpoint=entry.CHECKPOINT)

    def test_rgb_only_sample_and_teacher_forward_use_three_channels_and_no_vi_files(self):
        from test_doris_mechanism import MechanismTests
        from test_minimal_output import load_methods
        from tools import replay_doris_mechanisms as replay
        model=MechanismTests().model()
        model.enable_phy=model.enable_struct=model.enable_proto=model.enable_struct_edge=False
        model.model.conv=torch.nn.Conv2d(3,4,1)
        model.ema_model.conv=torch.nn.Conv2d(3,4,1)
        spec=importlib.util.spec_from_file_location('rgb_only_observation',ROOT/'mmseg/utils/mechanism.py')
        observation=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observation)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root=Path(directory)/'30'
            filename='3_1024_1280_1280_1536.tif'
            tif=root/'rgb_dir/val'/filename
            tif.parent.mkdir(parents=True)
            rgb=np.arange(256*256*3,dtype=np.uint8).reshape(256,256,3)
            Image.fromarray(rgb).save(tif)
            image=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255
            historical_norm=dict(mean=[0]*3,std=[255]*3,to_rgb=True)
            config=dict(name='doris_goji26RGB2goji30RGB_daformer_r101_no3vi',uda={},
                target_img_norm_cfg=dict(historical_norm,channel_order=[2,1,0,3,4,5]),
                data=dict(train=dict(target=dict(data_root=str(root),img_dir='rgb_dir/train',crop_pseudo_margins=[30]*4,
                    pipeline=[dict(type='Normalize',**historical_norm)])),
                    val=dict(data_root=str(root),img_dir='rgb_dir/val')))
            probe=observation.deterministic_probe_config(config,allow_daformer=True)
            mmcv=SimpleNamespace(imnormalize=lambda value,mean,std,to_rgb:
                ((value[...,::-1] if to_rgb else value).astype(np.float32)-mean)/std)
            normalize=load_methods('mmseg/datasets/pipelines/transforms.py','Normalize',
                ['__init__','__call__'],dict(np=np,mmcv=mmcv))
            norm_step=next(step for step in probe['dataset']['pipeline'] if step['type']=='Normalize')
            arguments={key:value for key,value in norm_step.items() if key!='type'}
            class Dataset:
                img_dir=str(tif.parent)
                img_infos=[dict(filename=filename)]
                def __getitem__(self,index):
                    loaded=normalize(**arguments)(dict(img=rgb[...,::-1].copy()))
                    tensor=torch.from_numpy(loaded['img'].transpose(2,0,1).copy())
                    return dict(img=SimpleNamespace(data=tensor),img_metas=SimpleNamespace(data=dict(
                        filename=str(tif),ori_shape=(256,256,3),img_norm_cfg=loaded['img_norm_cfg'])))
            modules={'mmseg.datasets':SimpleNamespace(build_dataset=lambda cfg:Dataset()),
                'mmseg.utils.mechanism':observation}
            report=replay.restore_response_state(model,model.state_dict(),15000)
            before=replay.state_digest(model.state_dict())
            with patch.dict('sys.modules',modules):
                sample,description=observation.load_probe_sample(probe,model)
                self.assertEqual(tuple(sample[0].shape),(1,3,256,256))
                self.assertEqual(tuple(sample[1].shape),(1,0,256,256))
                self.assertFalse(description['vi_available'])
                self.assertEqual(description['tensor_sha256'],hashlib.sha256(image.numpy().tobytes()).hexdigest())
                payload=replay.replay_payload(model,sample,probe,dict(iter=15000,sample_id=filename,
                    modules=report['requested_modules'],capture_mode='checkpoint_replay',original_shape=[256,256]),report)
            self.assertEqual(replay.state_digest(model.state_dict()),before)
            np.testing.assert_array_equal((payload['target_image']*255).round().astype(np.uint8),rgb)
            self.assertEqual(payload['w_conf'].shape,(256,256))
            self.assertEqual(int(payload['valid_mask'].sum()),196*196)
            self.assertNotIn('r_phy',payload)

    def test_daformer_entry_pins_its_checkpoint_and_reuses_baseline_replay(self):
        from tools import replay_doris_daformer_13000 as entry
        self.assertEqual(str(entry.CHECKPOINT).replace('\\','/'),
            '/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB_daformer_3vi/best_mIoU_iter_13000.pth')
        with patch.object(entry,'replay_baseline',return_value=0) as replay:
            self.assertEqual(entry.main([]),0)
        replay.assert_called_once_with([],default_checkpoint=entry.CHECKPOINT)

    def test_disabled_branches_keep_real_teacher_predictions_and_state(self):
        from test_doris_mechanism import MechanismTests
        from tools import replay_doris_mechanisms as replay
        model = MechanismTests().model()
        model.enable_phy = model.enable_struct = model.enable_proto = model.enable_struct_edge = False
        report = replay.restore_response_state(model,model.state_dict(),18000)
        spec = importlib.util.spec_from_file_location('baseline_observation',ROOT/'mmseg/utils/mechanism.py')
        observation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observation)
        sample = (torch.zeros(1,6,16,16),torch.ones(1,3,16,16),
            [dict(filename='SYNTHETIC.tif',img_norm_cfg=dict(mean=[0]*6,std=[255]*6))])
        metadata = dict(iter=18000,sample_id='SYNTHETIC.tif',synthetic_fixture=True,
            modules=report['requested_modules'],capture_mode='checkpoint_replay',original_shape=[16,16])
        before = replay.state_digest(model.state_dict())
        with patch.dict('sys.modules',{'mmseg.utils.mechanism':observation}):
            payload = replay.replay_payload(model,sample,dict(margins=[1]*4),metadata,report)
        self.assertEqual(replay.state_digest(model.state_dict()),before)
        self.assertIn('w_conf',payload)
        self.assertIn('pseudo_label',payload)
        for key in ('r_phy','r_struct','r_edge','delta_w_phy','delta_w_proto'):
            self.assertNotIn(key,payload)
        self.assertTrue(json.loads(str(payload['metadata']))['historical_state_complete'])

    def test_entry_uses_given_checkpoint_saved_config_and_unique_work_directory(self):
        from tools import replay_doris_baseline_18000 as entry
        self.assertEqual(str(entry.CHECKPOINT).replace('\\','/'),
            '/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB_hrda_3vi/iter_18000.pth')
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root = Path(directory)
            checkpoint = root/'iter_18000.pth'
            torch.save(dict(meta=dict(iter=17999,config='saved baseline config'),
                state_dict={'model.weight':torch.ones(1)}),checkpoint)
            tif = root/entry.FILENAME
            tif.write_bytes(b'synthetic presence fixture')
            config = SimpleNamespace(uda={})
            observation = SimpleNamespace(deterministic_probe_config=lambda cfg,**kwargs:dict(
                filename=entry.FILENAME,margins=[30]*4,expected_path=str(tif)))
            mmcv = SimpleNamespace(Config=SimpleNamespace(fromstring=lambda text,suffix:config))
            result = dict(status='complete',output=str(root/'node'),iter=18000)
            with patch.dict('sys.modules',{'mmcv':mmcv,'mmseg.utils.mechanism':observation}), \
                    patch.object(entry,'reference_path',return_value=tif), \
                    patch.object(torch.cuda,'is_available',return_value=True), \
                    patch.object(entry,'replay_node',return_value=result) as replay, \
                    patch.object(entry,'verify_output') as verify, \
                    patch.object(entry,'render_paper_result',return_value=[root/'node/paper/figure.png',root/'node/paper/figure.pdf']):
                self.assertEqual(entry.main(['--checkpoint',str(checkpoint)]),0)
                self.assertEqual(entry.main(['--checkpoint',str(checkpoint)]),0)
            self.assertEqual(replay.call_args.args[0],config)
            self.assertEqual(Path(replay.call_args.args[1]['path']),checkpoint.resolve())
            self.assertEqual(replay.call_args.args[1]['completed_iter'],18000)
            self.assertTrue(replay.call_args.kwargs['reference_evidence'])
            self.assertNotEqual(replay.call_args_list[0].args[2],replay.call_args_list[1].args[2])
            verify.assert_called_with(root/'node',tif,18000)
            self.assertEqual(len(list((root/'analysis/fixed_probe/replayed').glob('baseline_*/replay_report_18000.json'))),2)

    def test_render_only_avoids_cuda_and_checkpoint_loading(self):
        from tools import replay_doris_baseline_18000 as entry
        with patch.object(entry,'render_paper_result',return_value=[Path('figure.png')]) as render, \
                patch.object(torch,'load',side_effect=AssertionError('checkpoint load on redraw')), \
                patch.object(torch.cuda,'is_available',side_effect=AssertionError('CUDA on redraw')):
            self.assertEqual(entry.main(['--render-only','--responses','saved.npz']),0)
        render.assert_called_once_with(Path('saved.npz'),None)


@unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Matplotlib unavailable')
class BaselineFigureTests(unittest.TestCase):
    def test_five_panel_baseline_preserves_source_and_crops_all_maps(self):
        from test_doris_phy_evidence import PhysiologyFigureTests, ANNOTATION
        from tools.build_doris_mechanism_figures import render_baseline_paper, render_online_node
        from tools.check_doris_phy_semantics_7000 import SEMANTIC_PALETTE, vegetation_confusion_map
        maps = PhysiologyFigureTests().paper_fixture()
        for key in ('r_phy','w_before_phy','w_after_phy','delta_w_phy','m_phy'):
            maps.pop(key)
        metadata = json.loads(str(maps['metadata']))
        metadata.update(iter=18000,modules=dict(phy=False,struct=False,proto=False,edge=False),
            executed_modules=dict(phy=False,struct=False,proto=False,edge=False),
            historical_state_complete=True,reference_evidence=ANNOTATION)
        metadata.pop('phy_evidence')
        maps['metadata'] = np.array(json.dumps(metadata))
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root = Path(directory)
            source = root/'responses.npz'
            np.savez_compressed(source,**maps)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(render_online_node(maps,root,18000,render_timing='checkpoint_replay'),[])
            with Image.open(root/'spatial/iter_018000/w_conf.png') as image:
                self.assertEqual(image.size,(256,256))
            paths = render_baseline_paper(source)
            self.assertTrue(all(path.is_file() for path in paths))
            for path in (root/'paper/spatial').iterdir():
                if path.suffix in ('.png','.tif'):
                    with Image.open(path) as image:self.assertEqual(image.size,(196,196),path.name)
            with Image.open(root/'paper/spatial/phy_semantic_check.png') as image:
                expected = np.asarray(SEMANTIC_PALETTE,np.uint8)[vegetation_confusion_map(maps)[30:226,30:226]]
                np.testing.assert_array_equal(np.asarray(image),expected)
            with Image.open(root/'paper/spatial/target_image.tif') as image:
                expected = (maps['target_image'][30:226,30:226]*255).round().astype(np.uint8)
                np.testing.assert_array_equal(np.asarray(image),expected)
            info = json.loads((root/'paper/iter_018000_baseline_paper_metadata.json').read_text())
            self.assertEqual(info['display']['layout'],'1x5')
            self.assertEqual(info['source_npz_sha256'],digest)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            self.assertEqual(render_baseline_paper(source),paths)


if __name__=='__main__':
    unittest.main()
