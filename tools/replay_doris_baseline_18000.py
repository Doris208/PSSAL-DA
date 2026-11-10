"""Replay RGB-to-RGB Baselines with their historical three- or six-channel input."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools.doris_phy_evidence import reference_path
from tools.replay_doris_mechanisms import (
    checkpoint_iteration, checkpoint_state, file_digest, replay_node, state_digest, write_json)

CHECKPOINT = Path('/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB_hrda_3vi/iter_18000.pth')
FILENAME = '3_1024_1280_1280_1536.tif'
MODULES = dict(phy=False,struct=False,proto=False,edge=False)


def verify_output(folder, tif_path, expected_iter):
    import mmcv

    metadata = json.loads((folder/'replay_metadata.json').read_text(encoding='utf-8'))
    if (metadata['iter']!=expected_iter or metadata['capture_mode']!='checkpoint_replay'
            or metadata['modules']!=MODULES or metadata['executed_modules']!=MODULES
            or not metadata['historical_state_complete']):
        raise ValueError('Unexpected Baseline checkpoint recovery or enabled branches')
    reference = metadata['reference_evidence']
    if file_digest(reference['annotation_path']) != reference['annotation_sha256']:
        raise ValueError('Validation annotation changed during replay')
    rgb = mmcv.imread(str(tif_path),flag='color')[...,::-1]
    with np.load(folder/'responses.npz',allow_pickle=False) as maps:
        np.testing.assert_array_equal(np.rint(maps['target_image']*255).astype(np.uint8),rgb)
        if rgb.shape[:2]!=(256,256):
            raise ValueError('This comparison expects a native 256x256 target image')
        for key in ('w_conf','pseudo_label','reference_label','valid_mask'):
            if maps[key].shape!=(256,256):
                raise ValueError('Baseline map differs from the original TIF grid')
        expected = np.zeros((256,256),bool)
        expected[30:226,30:226] = True
        np.testing.assert_array_equal(maps['valid_mask'],expected)
        forbidden = ('r_phy','r_struct','r_edge','delta_w_phy','delta_w_proto',
            'w_before_phy','w_after_phy','w_before_proto','w_after_proto')
        if any(key in maps for key in forbidden):
            raise ValueError('Baseline replay produced a disabled branch response')


def render_paper_result(source, output=None):
    from tools.build_doris_mechanism_figures import render_baseline_paper

    return render_baseline_paper(source,output)


def main(argv=None, default_checkpoint=CHECKPOINT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=default_checkpoint,
        help='absolute path, or filename relative to the specified Baseline work directory')
    parser.add_argument('--render-only',action='store_true',help='render saved Baseline responses')
    parser.add_argument('--responses',type=Path,help='saved Baseline responses.npz for --render-only')
    parser.add_argument('--paper-out',type=Path,help='separate paper output directory')
    args = parser.parse_args(argv)
    if args.render_only:
        if not args.responses:
            parser.error('--render-only requires --responses')
        print('Figure:',render_paper_result(args.responses,args.paper_out)[0])
        return 0
    if args.responses:
        parser.error('--responses requires --render-only')
    from mmcv import Config
    from mmseg.utils.mechanism import deterministic_probe_config

    checkpoint = (default_checkpoint.parent/args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError('Baseline checkpoint is missing: '+str(checkpoint))
    digest = file_digest(checkpoint)
    saved = torch.load(checkpoint,map_location='cpu')
    saved_config = saved.get('meta',{}).get('config')
    if not saved_config:
        raise ValueError('Checkpoint lacks meta.config for historical Baseline verification')
    config = Config.fromstring(saved_config,'.py')
    modules = {name:bool(config.uda.get(key,{}).get('enable',False)) for name,key in
        (('phy','phy_cfg'),('struct','struct_cfg'),('proto','proto_cfg'),('edge','struct_edge_cfg'))}
    if modules!=MODULES:
        raise ValueError('The supplied checkpoint configuration enables a mechanism branch')
    probe = deterministic_probe_config(config,allow_daformer=True)
    if probe.get('filename')!=FILENAME or probe['margins']!=[30,30,30,30]:
        raise ValueError('Baseline target image or valid margins differ from the phy comparison')
    tif = Path(probe['expected_path'])
    for path in (tif,reference_path(config,FILENAME)):
        if not path.is_file():
            raise FileNotFoundError('Required Baseline replay input is missing: '+str(path))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the active Python environment')
    iteration = checkpoint_iteration(checkpoint,saved.get('meta',{}))
    state = checkpoint_state(saved)
    record = dict(path=str(checkpoint),status='readable',checkpoint_sha256=digest,
        state_sha256=state_digest(state),**iteration)
    del state,saved
    step = record['completed_iter']
    output = checkpoint.parent/'analysis/fixed_probe/replayed'/('baseline_{}_{}'.format(
        step,datetime.now().strftime('%m%d_%H%M%S_%f')))
    output.mkdir(parents=True,exist_ok=False)
    print('Replay output:',output)
    write_json(output/'checkpoint_audit.json',dict(data_source='checkpoint_replay',checkpoints=[record]))
    result = replay_node(config,record,output,torch.device('cuda:0'),reference_evidence=True)
    if result['status']!='complete':
        raise RuntimeError('Incomplete Baseline recovery: '+json.dumps(result['recovery']))
    folder = Path(result['output'])
    verify_output(folder,tif,step)
    papers = render_paper_result(folder/'responses.npz',args.paper_out)
    if file_digest(checkpoint)!=digest:
        raise RuntimeError('Checkpoint changed during Baseline replay')
    write_json(output/('replay_report_{}.json'.format(step)),dict(data_source='checkpoint_replay',
        config_source='checkpoint.meta.config',results=[result],paper_figures=[str(path) for path in papers]))
    print('PASS - Baseline replay, native-grid verification and paper export completed')
    print('Figure:',papers[0])
    print('Paper PDF:',papers[1])
    print('Aligned individual images:',papers[0].parent/'spatial')
    print('Cropped RGB TIF:',papers[0].parent/'spatial/target_image.tif')
    print('Confusion statistics:',papers[0].parent/'semantic_statistics.csv')
    print('Metadata:',folder/'replay_metadata.json')
    return 0


if __name__=='__main__':
    sys.exit(main())
