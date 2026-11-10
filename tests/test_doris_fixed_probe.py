"""Exact probe selection and native spatial exports, without a training run."""
import ast
from copy import deepcopy
import importlib.util
from io import BytesIO
import json
import os
import os.path as osp
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'outputs').mkdir(exist_ok=True)


def selection_api():
    tree = ast.parse((ROOT / 'mmseg/utils/mechanism.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                 t.id == 'FIXED_TARGET_PROBES' for t in n.targets) or
             isinstance(n, ast.FunctionDef) and n.name in
                 ('deterministic_probe_config', 'exact_probe_index')]
    scope = dict(deepcopy=deepcopy, Path=Path, re=re)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'probe_selection', 'exec'), scope)
    return scope


class FixedProbeSelectionTests(unittest.TestCase):
    def test_three_directions_use_exact_target_val_and_native_pipeline(self):
        api = selection_api()
        for direction, domain, view, filename in [
                ('goji30IRRG2goji26IRRG', '26', 'img_dir', '3_768_964_1024_1220.tif'),
                ('goji26IRRG2goji30RGB', '30', 'rgb_dir', '3_1536_640_1792_896.tif'),
                ('goji26RGB2goji30RGB', '30', 'rgb_dir', '3_1024_1280_1280_1536.tif')]:
            for suffix in ('_phy_gate', '_struct_gate', '_phy_struct_gate', '_ps_proto',
                           '_struct_edge', '_pbcr', '_proto_struct_edge'):
                cfg = dict(name='doris_' + direction + '_hrda_3vi' + suffix,
                    uda=dict(fixed_probe_index=99), crop_size=(256, 256),
                    target_img_norm_cfg=dict(mean=[0]*6, std=[1]*6),
                    data=dict(train=dict(target=dict(data_root='/data/' + domain,
                        img_dir=view + '/train', split='train.txt', pipeline=[])),
                        val=dict(data_root='/data/' + domain, img_dir=view + '/val')))
                before = deepcopy(cfg)
                probe = api['deterministic_probe_config'](cfg)
                self.assertEqual(probe['filename'], filename)
                self.assertEqual(probe['direction'], direction)
                self.assertEqual(probe['dataset']['img_dir'], view + '/val')
                self.assertEqual(probe['expected_path'],
                    str((Path('/data')/domain/view/'val'/filename).resolve()))
                self.assertIsNone(probe['dataset']['split'])
                self.assertFalse(any(s['type'] == 'Resize' or 'Random' in s['type']
                                     for s in probe['dataset']['pipeline']))
                self.assertEqual(cfg, before)

    def test_exact_match_fails_closed_on_missing_duplicate_or_escape(self):
        select = selection_api()['exact_probe_index']
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            root = Path(folder)
            (root / 'fixed.tif').write_bytes(b'fixture')
            dataset = SimpleNamespace(img_dir=folder, img_infos=[
                dict(filename='fixed.tif.extra'), dict(filename='fixed.tif')])
            self.assertEqual(select(dataset, 'fixed.tif'), 1)
            dataset.img_infos = [dict(filename='fixed.tif.extra')]
            with self.assertRaises(ValueError): select(dataset, 'fixed.tif')
            dataset.img_infos = [dict(filename='fixed.tif')]*2
            with self.assertRaises(ValueError): select(dataset, 'fixed.tif')
            dataset.img_infos = [dict(filename='../fixed.tif')]
            with self.assertRaises(ValueError): select(dataset, 'fixed.tif')

    def test_comparison_configs_keep_their_original_probe_policy(self):
        cfg=dict(name='doris_goji26RGB2goji30RGB_daformer_3vi', uda={},
                 target_img_norm_cfg={}, data=dict(train=dict(target=dict(pipeline=[]))))
        self.assertNotIn('filename', selection_api()['deterministic_probe_config'](cfg))

    def test_daformer_offline_replay_uses_exact_native_target_without_changing_config(self):
        cfg=dict(name='doris_goji26RGB2goji30RGB_daformer_3vi',uda={},
            target_img_norm_cfg=dict(mean=[0]*6,std=[1]*6),
            data=dict(train=dict(target=dict(data_root='/data/30',img_dir='rgb_dir/train',
                pipeline=[],crop_pseudo_margins=[30]*4,use_indices=True,use_vi=False)),
                val=dict(data_root='/data/30',img_dir='rgb_dir/val')))
        before=deepcopy(cfg)
        probe=selection_api()['deterministic_probe_config'](cfg,allow_daformer=True)
        self.assertEqual(probe['filename'],'3_1024_1280_1280_1536.tif')
        self.assertEqual(probe['margins'],[30]*4)
        self.assertEqual(probe['dataset']['img_dir'],'rgb_dir/val')
        self.assertTrue(probe['dataset']['use_vi'])
        self.assertFalse(any(step['type']=='Resize' for step in probe['dataset']['pipeline']))
        self.assertEqual(cfg,before)

    def test_r101_rgb_only_replay_collects_image_without_enabling_vi(self):
        cfg=dict(name='doris_goji26RGB2goji30RGB_daformer_r101_no3vi',
            uda=dict(input_channel_order=['R','G','B']),
            target_img_norm_cfg=dict(mean=[1,2,3],std=[4,5,6],to_rgb=True),
            data=dict(train=dict(target=dict(data_root='/data/30',img_dir='rgb_dir/train',
                pipeline=[],crop_pseudo_margins=[30]*4)),
                val=dict(data_root='/data/30',img_dir='rgb_dir/val')))
        before=deepcopy(cfg)
        probe=selection_api()['deterministic_probe_config'](cfg,allow_daformer=True)
        self.assertEqual(probe['filename'],'3_1024_1280_1280_1536.tif')
        self.assertFalse(probe['dataset'].get('use_vi',False))
        self.assertFalse(probe['dataset'].get('use_indices',False))
        self.assertEqual(probe['dataset']['pipeline'][-1]['keys'],['img'])
        self.assertEqual(probe['channel_order'],['R','G','B'])
        self.assertEqual(cfg,before)

    def test_rgb_replay_uses_historical_normalize_instead_of_inherited_six_channel_order(self):
        from test_minimal_output import load_methods
        mmcv=SimpleNamespace(imnormalize=lambda image,mean,std,to_rgb:
            ((image[...,::-1] if to_rgb else image).astype(np.float32)-mean)/std)
        normalize=load_methods('mmseg/datasets/pipelines/transforms.py','Normalize',
            ['__init__','__call__'],dict(np=np,mmcv=mmcv))
        rgb=np.arange(4*5*3,dtype=np.uint8).reshape(4,5,3)
        for conversion in (dict(to_rgb=True),dict(to_rgb=False,channel_order=[2,1,0])):
            with self.subTest(conversion=conversion):
                historical=dict(mean=[1,2,3],std=[4,5,6],**conversion)
                inherited=dict(mean=[1,2,3],std=[4,5,6],to_rgb=True,
                    channel_order=[2,1,0,3,4,5],vis_channel_order=[0,1,2])
                cfg=dict(name='doris_goji26RGB2goji30RGB_daformer_r101_no3vi',uda={},
                    target_img_norm_cfg=inherited,data=dict(
                        train=dict(target=dict(data_root='/data/30',img_dir='rgb_dir/train',
                            pipeline=[dict(type='Normalize',**historical)],crop_pseudo_margins=[30]*4)),
                        val=dict(data_root='/data/30',img_dir='rgb_dir/val')))
                before=deepcopy(cfg)
                with self.assertRaises(IndexError):
                    normalize(**inherited)(dict(img=rgb[...,::-1].copy()))
                probe=selection_api()['deterministic_probe_config'](cfg,allow_daformer=True)
                step=next(step for step in probe['dataset']['pipeline'] if step['type']=='Normalize')
                arguments={key:value for key,value in step.items() if key!='type'}
                self.assertEqual(arguments,historical)
                loaded=normalize(**arguments)(dict(img=rgb[...,::-1].copy()))
                expected=(rgb.astype(np.float32)-np.array([1,2,3],np.float32))/np.array([4,5,6],np.float32)
                np.testing.assert_array_equal(loaded['img'],expected)
                self.assertEqual(cfg,before)

    def test_baseline_vi_return_fixes_collect_and_preserves_six_channel_pixels(self):
        from test_minimal_output import load_methods
        tree=ast.parse((ROOT/'mmseg/datasets/pipelines/loading.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LoadGojiIndices')
        node.decorator_list=[]
        mmcv=SimpleNamespace(FileClient=lambda **kw:SimpleNamespace(get=lambda p:Path(p).read_bytes()),
            print_log=lambda *args:None)
        scope=dict(np=np,osp=osp,Image=Image,BytesIO=BytesIO,mmcv=mmcv)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'production_index_loader','exec'),scope)
        spec=importlib.util.spec_from_file_location('mmseg.vi_utils',ROOT/'mmseg/vi_utils.py')
        vi_module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vi_module)
        collect=load_methods('mmseg/datasets/pipelines/formating.py','Collect',['__init__','__call__'],
            dict(DC=lambda value,**kw:SimpleNamespace(data=value)))(keys=['img','vi_img'],meta_keys=[])
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory, \
                patch.dict(sys.modules,{'mmseg.vi_utils':vi_module}):
            rgb_only=dict(img=np.ones((4,5,3),np.float32),use_indices=False,use_vi=False)
            with patch.object(mmcv,'FileClient',side_effect=AssertionError('VI access for RGB-only input')):
                self.assertEqual(scope['LoadGojiIndices']()(rgb_only)['img'].shape,(4,5,3))
            root=Path(directory)
            names=['ndvi','ndre','cire']
            for name,value in zip(names,(-.2,.4,2.0)):
                (root/name).mkdir()
                Image.fromarray(np.full((4,5),value,np.float32)).save(root/name/'SYNTHETIC.tif')
            for normalize_input in (False,True):
                with self.subTest(vi_norm_for_input=normalize_input):
                    historical=dict(img=np.arange(4*5*3,dtype=np.float32).reshape(4,5,3),
                        img_info=dict(filename='SYNTHETIC.tif'),use_indices=True,use_vi=False,
                        vi_as_input=True,vi_norm_for_input=normalize_input,index_names=names,
                        index_prefixes={name:str(root/name) for name in names})
                    old=scope['LoadGojiIndices']()(deepcopy(historical))
                    self.assertEqual(old['img'].shape,(4,5,6))
                    with self.assertRaisesRegex(KeyError,'vi_img'):
                        collect(old)
                    replay=deepcopy(historical)
                    replay['use_vi']=True
                    new=scope['LoadGojiIndices']()(replay)
                    np.testing.assert_array_equal(new['img'],old['img'])
                    self.assertEqual(collect(new)['vi_img'].shape,(4,5,3))


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib unavailable')
class NativeSpatialExportTests(unittest.TestCase):
    def test_spatial_exports_have_exact_pixels_and_fixed_weight_scale(self):
        from tools import build_doris_mechanism_figures as plot
        x = np.tile(np.linspace(0, 1, 37, dtype=np.float32), (23, 1))
        maps = dict(target_image=np.dstack([x]*3), valid_mask=np.ones(x.shape, bool),
            w_before_proto=x, w_after_proto=x*.95, delta_w_proto=-x*.05,
            r_edge=x, edge_supervision=x*.006, proto_active=np.array(True),
            pseudo_label=(x*4).astype(np.uint8))
        maps['metadata'] = np.array(json.dumps(dict(iter=6500, sample_id='SYNTHETIC.tif',
            synthetic_fixture=True, original_shape=[23, 37], capture_mode='online_after_train_iter',
            data_source='fixed_probe_recomputed', modules=dict(phy=False, struct=False, proto=True, edge=True),
            states=dict(proto='active', edge='active'), edge_supervision_max=.006)))
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            files = plot.render_online_node(maps, folder, 6500)
            self.assertEqual(len(files), 1)
            with Image.open(files[0]) as png:
                self.assertGreaterEqual(png.info['dpi'][0], 299)
            spatial = Path(folder) / 'spatial/iter_006500'
            self.assertTrue((spatial / 'w_after_proto.png').is_file())
            for path in spatial.glob('*.png'):
                with Image.open(path) as png:
                    self.assertEqual(png.size, (37, 23))
            with Image.open(spatial / 'pseudo_label.png') as png:
                np.testing.assert_array_equal(np.array(png), maps['pseudo_label'])
            with Image.open(spatial / 'w_after_proto.png') as png:
                rgba = np.array(png)
            expected = plot.plt.get_cmap('viridis')(maps['w_after_proto'], bytes=True)
            np.testing.assert_array_equal(rgba, expected)
            meta = json.loads((spatial / 'metadata.json').read_text())
            self.assertAlmostEqual(meta['statistics']['w_after_proto']['mean'],
                                   float(maps['w_after_proto'].mean()), places=6)
            stamps={p:p.stat().st_mtime_ns for p in Path(folder).rglob('*') if p.is_file()}
            plot.render_online_node(maps, folder, 6500)
            self.assertEqual(stamps,{p:p.stat().st_mtime_ns for p in stamps})

    def test_unmapped_grid_and_nonfinite_responses_are_rejected(self):
        from tools.doris_probe_data import spatial_statistics
        with self.assertRaisesRegex(ValueError, 'Unmapped'):
            spatial_statistics(dict(valid_mask=np.ones((8,8),bool), r_edge=np.zeros((4,4))))
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            spatial_statistics(dict(valid_mask=np.ones((8,8),bool), r_edge=np.full((8,8),np.nan)))


@unittest.skipUnless(os.environ.get('DORIS_PROBE_DATA_ROOT') and importlib.util.find_spec('torch'),
                     'Set DORIS_PROBE_DATA_ROOT and use a Torch/OpenCV interpreter for real TIF checks')
class RealTargetInputTests(unittest.TestCase):
    def test_three_real_tifs_through_loading_normalization_and_readonly_branches(self):
        import cv2
        import torch
        import test_minimal_output as helpers
        from test_doris_mechanism import MechanismTests
        spec=importlib.util.spec_from_file_location('real_probe',ROOT/'mmseg/utils/mechanism.py')
        obs=importlib.util.module_from_spec(spec);spec.loader.exec_module(obs)
        vi_spec=importlib.util.spec_from_file_location('mmseg.vi_utils',ROOT/'mmseg/vi_utils.py')
        vi_module=importlib.util.module_from_spec(vi_spec);vi_spec.loader.exec_module(vi_module)

        # Execute production transform classes. Only the absent MMCV file/image
        # adapters and registry are replaced, using its OpenCV BGR convention.
        mmcv=SimpleNamespace(FileClient=lambda **kw:SimpleNamespace(get=lambda p:Path(p).read_bytes()),
            imfrombytes=lambda b,**kw:cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR),
            imnormalize=lambda x,mean,std,to_rgb:(x.astype(np.float32)-mean)/std,
            print_log=lambda *args:None)
        scope=dict(np=np,osp=osp,Image=Image,BytesIO=BytesIO,mmcv=mmcv,
            PIPELINES=SimpleNamespace(register_module=lambda:lambda cls:cls))
        for file,names in [('mmseg/datasets/pipelines/loading.py',('LoadImageFromFile','LoadGojiIndices')),
                           ('mmseg/datasets/pipelines/transforms.py',('Normalize',))]:
            tree=ast.parse((ROOT/file).read_text(encoding='utf-8'))
            nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),file,'exec'),scope)
        data_root=Path(os.environ['DORIS_PROBE_DATA_ROOT'])
        for direction,(domain,view,filename) in obs.FIXED_TARGET_PROBES.items():
            cfg=helpers.static_config('configs/hrda/doris/'+direction+'/doris_'+direction+
                                      '_hrda_3vi_proto_struct_edge.py')
            cfg['data']['train']['target']['data_root']=str(data_root/domain)
            cfg['data']['val']['data_root']=str(data_root/domain)
            probe=obs.deterministic_probe_config(cfg)
            dataset_cfg=probe['dataset']
            image_dir=data_root/domain/view/'val'
            path=image_dir/filename
            self.assertTrue(path.is_file(),str(path))
            dataset=SimpleNamespace(img_dir=str(image_dir),img_infos=[dict(filename=filename)])

            def build(config):
                self.assertEqual(config,dataset_cfg)
                class Dataset:
                    img_dir=dataset.img_dir
                    img_infos=dataset.img_infos
                    def __getitem__(self,index):
                        names=dataset_cfg['index_names']
                        results=dict(img_prefix=self.img_dir,img_info=self.img_infos[index],
                            use_indices=True,use_vi=True,vi_as_input=dataset_cfg.get('vi_as_input',True),
                            index_names=names,vi_names=dataset_cfg.get('vi_names',names),
                            vi_norm_cfg=dataset_cfg['vi_norm_cfg'],
                            vi_norm_for_input=dataset_cfg.get('vi_norm_for_input',False),
                            index_prefixes={n:str(data_root/domain/(n+'_dir')/'val') for n in names})
                        for step in config['pipeline'][:3]:
                            kwargs={k:v for k,v in step.items() if k!='type'}
                            results=scope[step['type']](**kwargs)(results)
                        return dict(img=SimpleNamespace(data=torch.from_numpy(
                            np.ascontiguousarray(results['img'].transpose(2,0,1)))),
                            vi_img=SimpleNamespace(data=torch.from_numpy(
                                np.ascontiguousarray(results['vi_img'].transpose(2,0,1)))),
                            img_metas=SimpleNamespace(data={k:results[k] for k in
                                ('filename','ori_shape','img_norm_cfg')}))
                return Dataset()

            with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder, patch.dict(sys.modules,
                    {'mmseg.datasets':SimpleNamespace(build_dataset=build),'mmseg.vi_utils':vi_module}):
                old_sample=Path(folder)/'analysis/fixed_probe/sample_manifest.json'
                old_sample.parent.mkdir(parents=True)
                old_sample.write_text('{"sample_id":"old_scene.tif"}')
                previous=old_sample.read_bytes()
                runner=SimpleNamespace(work_dir=folder,iter=0,meta={},mechanism_config=cfg,
                    logger=SimpleNamespace(info=lambda *args:None))
                obj=obs.MechanismCollector(runner,MechanismTests().model())
                self.assertIn(Path(filename).stem,str(obj.root))
                self.assertEqual(old_sample.read_bytes(),previous)
                first=[v.clone() for v in obj.sample[:2]]
                resumed=obs.MechanismCollector(runner,obj.model)
                self.assertEqual(resumed.root,obj.root)
                obj=resumed
                image,vi,metas=obj.sample
                self.assertEqual(tuple(image.shape),(1,6,256,256))
                self.assertEqual(tuple(vi.shape),(1,3,256,256))
                for a,b in zip(first,obj.sample[:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
                original=cv2.imread(str(path),cv2.IMREAD_COLOR)
                norm=cfg['target_img_norm_cfg']
                restored=image[0,:3].numpy().transpose(1,2,0)*np.array(norm['std'][:3])+np.array(norm['mean'][:3])
                np.testing.assert_allclose(restored,original[...,norm['channel_order'][:3]],atol=2e-5)
                state=obs.model_digest(obj.model);rng=torch.get_rng_state().clone()
                extract=obj.model.get_model().extract_feat
                obj.model.get_model().extract_feat=lambda x:[torch.nn.functional.avg_pool2d(extract(x)[0],64)]
                for step in (3000,6500):
                    obj.model.local_iter=step
                    maps=obs.probe_arrays(obj.model,image,vi,metas,torch.ones(1,1,256,256))
                    self.assertTrue(all(np.isfinite(v).all() for v in maps.values()))
                    for key,value in maps.items():
                        if value.ndim==2:self.assertEqual(value.shape,(256,256),key)
                    np.testing.assert_array_equal(maps['delta_w_proto'],maps['w_t_proto']-maps['w_pre_proto'])
                    if step==3000:np.testing.assert_array_equal(maps['w_t_proto'],maps['w_pre_proto'])
                    self.assertEqual(state,obs.model_digest(obj.model))
                    self.assertEqual(maps['prototype_feature_shape'].tolist(),[4,4])
                    torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
                print('REAL TIF / synthetic backbone:',direction,str(path),'input',tuple(image.shape),
                      'VI',tuple(vi.shape),'maps',maps['r_edge'].shape)
                plot_python = os.environ.get('DORIS_PLOT_PYTHON')
                if plot_python:
                    payload = dict(maps, target_image=(restored/255).astype(np.float32),
                        w_before_phy=maps['w_base'], w_after_phy=maps['w_t_phy'],
                        w_before_proto=maps['w_pre_proto'], w_after_proto=maps['w_t_proto'])
                    payload['metadata'] = np.array(json.dumps(dict(iter=6500, sample_id=filename,
                        synthetic_fixture=True, original_shape=[256,256],
                        capture_mode='synthetic_backbone_real_input_test',
                        data_source='fixed_probe_recomputed', modules=obs.module_flags(obj.model),
                        states=obs.module_states(obj.model), edge_supervision_max=.006)))
                    source = Path(folder)/'synthetic_backbone_real_input.npz'
                    np.savez_compressed(source, **payload)
                    output = Path(folder)/'rendered'
                    code = ('import sys; from tools.build_doris_mechanism_figures import '
                            'read_node, render_online_node; '
                            'render_online_node(read_node(sys.argv[1]),sys.argv[2],6500)')
                    subprocess.run([plot_python, '-c', code, str(source), str(output)],
                                   cwd=ROOT, check=True, capture_output=True, text=True)
                    spatial = output/'spatial/iter_006500'
                    self.assertEqual(len(list(output.glob('*.png'))), 2)
                    for png in spatial.glob('*.png'):
                        with Image.open(png) as exported:
                            self.assertEqual(exported.size, (256,256))
                    with Image.open(spatial/'target_image.png') as exported:
                        np.testing.assert_array_equal(np.array(exported),
                            original[...,norm['channel_order'][:3]])


if __name__ == '__main__':
    unittest.main()
