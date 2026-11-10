"""Historical-state replay tests with explicit small-model fixtures."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from test_minimal_output import ROOT, torch


@unittest.skipIf(torch is None, 'Torch unavailable')
class CheckpointReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools import replay_doris_mechanisms
        cls.replay = replay_doris_mechanisms
        spec = importlib.util.spec_from_file_location('replay_test_observation', ROOT/'mmseg/utils/mechanism.py')
        observation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observation)
        cls.imports = patch.dict('sys.modules', {'mmseg.utils.mechanism': observation})
        cls.imports.start()
        cls.addClassCleanup(cls.imports.stop)

    def model(self):
        from test_doris_mechanism import MechanismTests
        return MechanismTests().model()

    def test_iteration_uses_completed_filename_and_rejects_ambiguous_latest(self):
        resolve = self.replay.checkpoint_iteration
        self.assertEqual(resolve(Path('iter_6000.pth'), {'iter': 5999})['completed_iter'], 6000)
        self.assertEqual(resolve(Path('best_mIoU_iter_6000.pth'), {'iter': 6000})['completed_iter'], 6000)
        with self.assertRaises(ValueError):
            resolve(Path('iter_6000.pth'), {'iter': 20000})
        with self.assertRaises(ValueError):
            resolve(Path('latest.pth'), {'iter': 19999})

    def test_discovery_reports_corruption_and_pilot_uses_two_actual_nodes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            root = Path(folder)
            for step in (6000, 12000, 20000):
                torch.save(dict(meta=dict(iter=step-1), state_dict={'model.weight': torch.tensor([step])}),
                           root/('iter_%d.pth' % step))
            (root/'latest.pth').write_bytes((root/'iter_20000.pth').read_bytes())
            (root/'iter_15000.pth').write_bytes(b'PK\x03\x04broken')
            records = self.replay.audit_checkpoints(root)
            self.assertEqual(len(records), 5)
            self.assertEqual(sum(r['status'] == 'unreadable' for r in records), 1)
            selected = self.replay.select_checkpoints(records, pilot=True)
            self.assertEqual([r['completed_iter'] for r in selected], [12000, 20000])
            with self.assertRaises(ValueError):
                self.replay.select_checkpoints(records, iterations=[10000])

    def test_missing_prototype_keeps_mparc_and_boundary_but_hides_proto_weights(self):
        model = self.model()
        state = copy.deepcopy(model.state_dict())
        state.pop('proto_gate.prototypes')
        report = self.replay.restore_response_state(model, state, 6500)
        self.assertEqual(report['executed_modules'], dict(phy=True, struct=True, proto=False, edge=True))
        payload = self.payload(model, report)
        self.assertIn('r_struct', payload)
        self.assertIn('r_edge', payload)
        self.assertNotIn('w_before_proto', payload)
        self.assertNotIn('w_after_proto', payload)
        self.assertNotIn('proto_active', payload)
        self.assertFalse(json.loads(str(payload['metadata']))['historical_state_complete'])

    def test_missing_upstream_head_never_relabels_partial_weights_as_full_weights(self):
        model = self.model()
        state = {k:v for k,v in model.state_dict().items() if not k.startswith('phy_head.')}
        report = self.replay.restore_response_state(model, state, 6500)
        payload = self.payload(model, report)
        self.assertIn('r_struct', payload)
        self.assertIn('r_edge', payload)
        self.assertNotIn('w_before_struct', payload)
        self.assertNotIn('w_after_struct', payload)
        self.assertNotIn('w_after_proto', payload)
        self.assertNotIn('w_final', json.loads(str(payload['metadata']))['statistics'])

    def test_missing_teacher_blocks_response_computation(self):
        model = self.model()
        state = {k:v for k,v in model.state_dict().items() if not k.startswith('ema_model.')}
        report = self.replay.restore_response_state(model, state, 6500)
        self.assertFalse(report['teacher_complete'])
        with self.assertRaisesRegex(ValueError, 'Teacher'):
            self.payload(model, report)

    def test_fused_prototypes_can_recover_weights_without_claiming_missing_domain_history(self):
        model = self.model()
        state = copy.deepcopy(model.state_dict())
        state.pop('proto_gate.target_prototypes')
        state.pop('proto_gate.target_valid_proto')
        report = self.replay.restore_response_state(model, state, 6500)
        self.assertTrue(report['executed_modules']['proto'])
        self.assertFalse(report['historical_state_complete'])
        payload = self.payload(model, report)
        self.assertIn('w_after_proto', payload)

    def test_config_comparison_detects_algorithm_changes_and_ignores_output_paths(self):
        from test_minimal_output import static_config
        cfg = static_config('configs/hrda/doris/goji26RGB2goji30RGB/'
                            'doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py')
        other = copy.deepcopy(cfg)
        other['uda']['fixed_probe_index'] = 123
        other['model']['train_cfg']['work_dir'] = '/new/output'
        self.assertEqual(self.replay.inference_signature(cfg), self.replay.inference_signature(other))
        other['uda']['proto_cfg']['weight_start'] = 0
        self.assertNotEqual(self.replay.inference_signature(cfg), self.replay.inference_signature(other))

    def test_phy_entry_selects_exact_best_file_and_runs_verification(self):
        from tools import replay_doris_phy_7000 as entry
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root = Path(directory)
            work = root/'work'
            work.mkdir()
            best = work/'best_mIoU_iter_7000.pth'
            torch.save(dict(state_dict={'model.weight':torch.tensor([7.])}, meta=dict(iter=6999)), best)
            torch.save(dict(state_dict={'model.weight':torch.tensor([99.])}, meta=dict(iter=6999)),
                       work/'iter_7000.pth')
            late = work/'iter_10000.pth'
            torch.save(dict(state_dict={'model.weight':torch.tensor([10.])}, meta=dict(iter=9999)),late)
            tif = root/'30/rgb_dir/val'/entry.FILENAME
            names = ['ndvi', 'ndre', 'cire']
            for path in [tif] + [root/'30'/(name+'_dir')/'val'/entry.FILENAME for name in names]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'synthetic input presence fixture')
            annotation = root/'30/ann_dir/val'/entry.FILENAME
            annotation.parent.mkdir(parents=True)
            annotation.write_bytes(b'synthetic annotation presence fixture')
            config = SimpleNamespace(work_dir=str(work), uda={
                key:dict(enable=key=='phy_cfg') for key in
                ('phy_cfg', 'struct_cfg', 'proto_cfg', 'struct_edge_cfg')},
                data=SimpleNamespace(train=SimpleNamespace(target=SimpleNamespace(
                    index_names=names, get=lambda key, default:default)),
                    val=SimpleNamespace(img_dir='rgb_dir/val', data_root=str(root/'30'))))
            result = dict(status='complete', output=str(root/'result'), iter=7000)
            observation = sys.modules['mmseg.utils.mechanism']
            with patch.dict(sys.modules, {'mmcv':SimpleNamespace(Config=SimpleNamespace(
                    fromfile=lambda path:config))}), patch.object(observation, 'deterministic_probe_config',
                    return_value=dict(filename=entry.FILENAME, expected_path=str(tif))), \
                    patch.object(entry, 'reference_path', return_value=annotation), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(entry, 'replay_node', return_value=result) as replay, \
                    patch.object(entry, 'render_paper_result',return_value=[
                        root/'result/paper/iter_007000_phys_paper.png',
                        root/'result/paper/iter_007000_phys_paper.pdf']) as paper, \
                    patch.object(entry, 'verify_output') as verify:
                self.assertEqual(entry.main([]), 0)
                self.assertEqual(entry.main([]), 0)
                later_result = dict(status='complete',output=str(root/'later_result'),iter=10000)
                replay.return_value = later_result
                paper.return_value = [root/'later_result/paper/iter_010000_phys_paper.png',
                    root/'later_result/paper/iter_010000_phys_paper.pdf']
                self.assertEqual(entry.main(['--checkpoint','iter_10000.pth']),0)
            self.assertEqual(Path(replay.call_args_list[0].args[1]['path']),best.resolve())
            self.assertEqual(Path(replay.call_args_list[2].args[1]['path']),late.resolve())
            self.assertTrue(all(call.kwargs['phy_evidence'] for call in replay.call_args_list))
            self.assertNotEqual(replay.call_args_list[0].args[2],replay.call_args_list[1].args[2])
            self.assertEqual(verify.call_args_list[0].args,(root/'result',tif,7000))
            self.assertEqual(verify.call_args_list[2].args,(root/'later_result',tif,10000))
            self.assertEqual(paper.call_args_list[0].args,(root/'result/responses.npz',None))
            reports = list((work/'analysis/fixed_probe/replayed').glob('phy_*/replay_report_*.json'))
            self.assertEqual(len(reports),3)
            self.assertEqual(sorted(json.loads(path.read_text())['results'][0]['iter'] for path in reports),
                [7000,7000,10000])

    def test_phy_paper_export_covers_weights_from_later_checkpoints(self):
        from tools import replay_doris_phy_7000 as entry
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root = Path(directory)
            source = root/'responses.npz'
            np.savez(source,valid_mask=np.ones((2,2),bool),delta_w_phy=np.full((2,2),-.34,np.float32))
            from unittest.mock import Mock
            render = Mock(return_value=[root/'figure.png',root/'figure.pdf'])
            with patch.dict(sys.modules,{'tools.build_doris_mechanism_figures':SimpleNamespace(render_phy_paper=render)}):
                entry.render_paper_result(source)
            self.assertEqual(render.call_args.kwargs['weight_range'],(0,1))
            self.assertAlmostEqual(render.call_args.kwargs['delta_range'][0],-.35)

    def test_phy_entry_render_only_reads_saved_arrays_without_model_or_cuda(self):
        from tools import replay_doris_phy_7000 as entry
        with patch.object(entry,'render_paper_result') as paper, \
                patch.object(entry,'replay_node',side_effect=AssertionError('Model inference on redraw')), \
                patch.object(torch.cuda,'is_available',side_effect=AssertionError('CUDA check on redraw')):
            self.assertEqual(entry.main(['--render-only','--responses','saved.npz']),0)
        paper.assert_called_once_with(Path('saved.npz'),None)

    def test_phy_entry_verifies_rgb_grid_weights_and_rejects_extra_branch(self):
        from tools import replay_doris_phy_7000 as entry
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            folder = Path(directory)
            rgb = np.arange(16*16*3, dtype=np.uint8).reshape(16,16,3)
            tif = folder/'SYNTHETIC.tif'
            Image.fromarray(rgb).save(tif)
            weight = np.full((16,16), .5, np.float32)
            maps = dict(target_image=rgb.astype(np.float32)/255,
                w_conf=weight+.3, r_phy=weight, w_before_phy=weight, w_after_phy=weight*.8,
                delta_w_phy=weight*.8-weight, m_phy=np.ones(weight.shape,bool),
                valid_mask=np.ones(weight.shape,bool), reference_label=np.zeros(weight.shape,np.uint8))
            metadata = dict(iter=7000, capture_mode='checkpoint_replay',
                modules=entry.MODULES, executed_modules=entry.MODULES, synthetic_fixture=True,
                phy_evidence=dict(annotation_path=str(tif), annotation_sha256=entry.file_digest(tif)))
            (folder/'replay_metadata.json').write_text(json.dumps(metadata))
            np.savez(folder/'responses.npz', **maps)
            spatial = folder/'spatial/iter_007000'
            spatial.mkdir(parents=True)
            for name in maps:
                Image.fromarray(rgb).save(spatial/(name+'.png'))
            Image.fromarray(rgb).save(folder/'iter_007000_phys.png')
            mmcv = SimpleNamespace(imread=lambda path, flag:np.array(Image.open(path))[...,::-1])
            with patch.dict(sys.modules, {'mmcv':mmcv}):
                entry.verify_output(folder, tif)
                maps['r_struct'] = weight
                np.savez(folder/'responses.npz', **maps)
                with self.assertRaisesRegex(ValueError, 'Disabled branch'):
                    entry.verify_output(folder, tif)

    def test_phy_payload_keeps_actual_teacher_confidence_separate_from_dacs_weight(self):
        model = self.model()
        model.enable_struct = model.enable_proto = model.enable_struct_edge = False
        report = self.replay.restore_response_state(model, model.state_dict(), 7000)
        logits = []
        hook = model.ema_model.classifier.register_forward_hook(
            lambda module, inputs, output:logits.append(output.detach().clone()))
        try:
            payload = self.payload(model, report)
        finally:
            hook.remove()
        expected = logits[0].softmax(1).max(1).values[0].numpy()
        np.testing.assert_array_equal(payload['w_conf'], expected)
        self.assertFalse(np.array_equal(payload['w_conf'], payload['w_before_phy']))

    def payload(self, model, report):
        sample = (torch.randn(1,6,16,16), torch.rand(1,3,16,16),
                  [dict(filename='SYNTHETIC.tif', img_norm_cfg=dict(mean=[0]*6,std=[255]*6))])
        meta = dict(iter=6500, sample_id='SYNTHETIC.tif', synthetic_fixture=True,
                    modules=dict(phy=True,struct=True,proto=True,edge=True),
                    data_source='checkpoint_replay', capture_mode='checkpoint_replay',
                    original_shape=[16,16])
        return self.replay.replay_payload(model, sample, dict(margins=[1]*4), meta, report)

    def test_two_checkpoint_states_produce_readonly_traceable_responses(self):
        model = self.model()
        original_flags = (model.enable_phy, model.enable_struct, model.enable_proto, model.enable_struct_edge)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            for step in (6500, 20000):
                checkpoint = Path(folder)/('iter_%d.pth' % step)
                torch.save(dict(state_dict=model.state_dict(),meta=dict(iter=step-1)),checkpoint)
                before_file = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                state = torch.load(checkpoint,map_location='cpu')['state_dict']
                report = self.replay.restore_response_state(model,state,step)
                before = {k:v.clone() for k,v in model.state_dict().items()}
                rng = torch.get_rng_state().clone()
                sample = (torch.zeros(1,6,16,16), torch.ones(1,3,16,16),
                    [dict(filename='SYNTHETIC.tif',img_norm_cfg=dict(mean=[0]*6,std=[255]*6))])
                metadata = dict(iter=step,sample_id='SYNTHETIC.tif',synthetic_fixture=True,
                    modules=dict(phy=True,struct=True,proto=True,edge=True),
                    data_source='checkpoint_replay',capture_mode='checkpoint_replay',original_shape=[16,16])
                payload = self.replay.replay_payload(model,sample,dict(margins=[1]*4),metadata,report)
                np.testing.assert_array_equal(payload['delta_w_proto'],
                    payload['w_after_proto']-payload['w_before_proto'])
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,before[key],rtol=0,atol=0)
                torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
                self.assertEqual(before_file,hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                self.assertEqual(original_flags,
                    (model.enable_phy,model.enable_struct,model.enable_proto,model.enable_struct_edge))

    def test_replay_node_writes_two_traceable_plates_and_native_maps(self):
        # Exercise orchestration with a synthetic backbone, Config and sample;
        # real shared branch forwards and PNG export still execute.
        if not os.environ.get('DORIS_PLOT_PYTHON') and not importlib.util.find_spec('matplotlib'):
            self.skipTest('Set DORIS_PLOT_PYTHON to an existing Matplotlib interpreter')
        from test_minimal_output import static_config

        class Config(dict):
            __setattr__ = dict.__setitem__

            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name)

            @classmethod
            def wrap(cls, value):
                if isinstance(value, dict):
                    return cls({k:cls.wrap(v) for k,v in value.items()})
                return value

            @classmethod
            def fromstring(cls, value, extension):
                return cls.wrap(json.loads(value))

        cfg = Config.wrap(static_config('configs/hrda/doris/goji26RGB2goji30RGB/'
            'doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py'))
        observation = sys.modules['mmseg.utils.mechanism']

        def render(maps, output, completed, **kwargs):
            code = ('import sys;from pathlib import Path;'
                'from tools.build_doris_mechanism_figures import read_node,render_online_node;'
                'render_online_node(read_node(Path(sys.argv[1])/"responses.npz"),sys.argv[1],'
                'int(sys.argv[2]),array_sha256=sys.argv[3],render_timing="checkpoint_replay")')
            subprocess.run([os.environ.get('DORIS_PLOT_PYTHON',sys.executable), '-c', code,
                str(output),str(completed),kwargs['array_sha256']],cwd=ROOT,check=True,capture_output=True)
            return sorted(Path(output).glob('iter_*.png'))

        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root = Path(directory)
            target = root/'SYNTHETIC.tif'
            target.write_bytes(b'synthetic sample loader fixture, not experiment data')
            sample = (torch.zeros(1,6,16,16),torch.ones(1,3,16,16),
                [dict(img_norm_cfg=dict(mean=[0]*6,std=[255]*6))])
            sample_meta = dict(target_path=str(target),tensor_sha256='synthetic',
                original_shape=[16,16],spatial_shape=[16,16],synthetic_fixture=True)
            for step in (6500,20000):
                torch.save(dict(state_dict=self.model().state_dict(),
                    meta=dict(iter=step-1,config=json.dumps(cfg))),root/('iter_%d.pth'%step))
            rows = self.replay.select_checkpoints(self.replay.audit_checkpoints(root),pilot=True)
            modules = {'mmcv':SimpleNamespace(Config=Config),
                'mmseg.models.builder':SimpleNamespace(build_train_model=lambda cfg:self.model()),
                'tools.build_doris_mechanism_figures':SimpleNamespace(render_online_node=render)}
            from PIL import Image
            with patch.dict(sys.modules,modules), patch.object(observation,'load_probe_sample',
                    return_value=(sample,sample_meta)):
                for row in rows:
                    result = self.replay.replay_node(cfg,row,root/'replayed',torch.device('cpu'))
                    self.assertEqual(result['status'],'complete')
                    self.assertEqual(len(result['figures']),2)
                    output = Path(result['output'])
                    meta = json.loads((output/'replay_metadata.json').read_text())
                    self.assertEqual(meta['checkpoint_sha256'],row['checkpoint_sha256'])
                    self.assertEqual(meta['capture_mode'],'checkpoint_replay')
                    self.assertFalse(meta['ema_update_executed'])
                    for path in output.glob('spatial/**/*.png'):
                        with Image.open(path) as image:
                            self.assertEqual(image.size,(16,16))
                            self.assertEqual(json.loads(image.info['Description'])['checkpoint_sha256'],
                                             row['checkpoint_sha256'])
                    self.assertEqual(self.replay.file_digest(row['path']),row['checkpoint_sha256'])

                from test_doris_phy_evidence import annotation_loader, ANNOTATION
                annotation = root/'30/ann_dir/val/3_1024_1280_1280_1536.tif'
                annotation.parent.mkdir(parents=True)
                Image.fromarray(np.ones((16,16),np.uint8)).save(annotation)
                annotation_modules = {
                    'mmseg.datasets.pipelines':SimpleNamespace(LoadAnnotations=annotation_loader()),
                    'mmseg.datasets.goji_meta':SimpleNamespace(
                        GOJI_CLASSES=ANNOTATION['class_names'],GOJI_PALETTE=ANNOTATION['palette'])}
                with patch.dict(sys.modules,annotation_modules):
                    result = self.replay.replay_node(cfg,rows[-1],root/'with_reference',
                        torch.device('cpu'),data_root=root/'30',phy_evidence=True)
                output = Path(result['output'])
                self.assertTrue((output/'phy_class_statistics.json').is_file())
                self.assertTrue((output/'phy_class_statistics.csv').is_file())
                with np.load(output/'responses.npz') as maps:
                    np.testing.assert_array_equal(maps['reference_label'],np.zeros((16,16),np.uint8))


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib unavailable')
class PartialReplayFigureTests(unittest.TestCase):
    def test_missing_prototype_panel_is_na_and_replay_source_is_visible(self):
        from tools.build_doris_mechanism_figures import online_plate, plt
        x = np.ones((16,16),np.float32)*.4
        maps = dict(target_image=np.dstack([x]*3),valid_mask=np.ones(x.shape,bool),
                    r_phy=x,r_struct=x,w_before_phy=x,w_after_struct=x,r_edge=x,
                    edge_supervision=x*.006)
        maps['metadata'] = np.array(json.dumps(dict(iter=6500,sample_id='SYNTHETIC.tif',
            modules=dict(phy=True,struct=True,proto=True,edge=True),states=dict(proto='unavailable'),
            capture_mode='checkpoint_replay',data_source='checkpoint_replay',
            historical_state_complete=False,synthetic_fixture=True,edge_supervision_max=.006)))
        fig = online_plate([maps],'pbcr')
        try:
            self.assertEqual(sum(t.get_text()=='N/A' for ax in fig.axes for t in ax.texts),2)
            self.assertIn('checkpoint_replay',fig._supxlabel.get_text())
            self.assertIn('PARTIAL',fig._supxlabel.get_text())
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
