"""Replay RGB-to-RGB physiology checkpoints, defaulting to Best mIoU Iter 7000."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.doris_probe_data import spatial_statistics
from tools.doris_phy_evidence import reference_path
from tools.replay_doris_mechanisms import (
    checkpoint_iteration, checkpoint_state, file_digest,
    replay_node, state_digest, write_json,
)

CONFIG = ROOT / 'configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py'
FILENAME = '3_1024_1280_1280_1536.tif'
MODULES = dict(phy=True, struct=False, proto=False, edge=False)


def verify_output(folder, tif_path, expected_iter=7000):
    import mmcv
    from PIL import Image

    metadata = json.loads((folder / 'replay_metadata.json').read_text(encoding='utf-8'))
    if (metadata['iter'] != expected_iter or metadata['capture_mode'] != 'checkpoint_replay' or
            metadata['modules'] != MODULES or metadata['executed_modules'] != MODULES):
        raise ValueError('Unexpected replay iteration, source or enabled modules')
    evidence = metadata['phy_evidence']
    if file_digest(evidence['annotation_path']) != evidence['annotation_sha256']:
        raise ValueError('Validation annotation differs from the saved evidence provenance')
    raw_rgb = mmcv.imread(str(tif_path), flag='color')[..., ::-1]
    with np.load(folder / 'responses.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            np.rint(arrays['target_image'] * 255).astype(np.uint8), raw_rgb)
        if arrays['valid_mask'].shape != raw_rgb.shape[:2]:
            raise ValueError('Response grid differs from the original TIF')
        statistics = spatial_statistics(arrays)
        for name in ('r_struct', 'r_edge', 'w_before_proto', 'w_after_proto', 'delta_w_proto'):
            if name in arrays.files:
                raise ValueError('Disabled branch produced a response: ' + name)
        np.testing.assert_allclose(arrays['delta_w_phy'],
            arrays['w_after_phy'] - arrays['w_before_phy'], rtol=0, atol=1e-6)
        valid = arrays['valid_mask'].astype(bool)
        if not valid.any() or np.any(arrays['w_after_phy'] > arrays['w_before_phy'] + 1e-6):
            raise ValueError('Empty valid region or invalid physiology weight increase')
        required = ('target_image', 'w_conf', 'r_phy', 'w_before_phy', 'w_after_phy',
                    'delta_w_phy', 'm_phy', 'valid_mask', 'reference_label')
        for name in required:
            path = folder / 'spatial' / ('iter_{:06d}'.format(expected_iter)) / (name + '.png')
            with Image.open(path) as image:
                if image.size != raw_rgb.shape[1::-1]:
                    raise ValueError('Spatial PNG dimensions differ from the TIF: ' + str(path))
                image.verify()
        print(json.dumps({name: statistics[name] for name in required[1:6]}, indent=2))
        print('Valid physiology-class pixels:', int((arrays['m_phy'] & valid).sum()))
    with Image.open(folder / ('iter_{:06d}_phys.png'.format(expected_iter))) as image:
        image.verify()


def render_paper_result(source, output=None):
    from tools.build_doris_mechanism_figures import render_phy_paper

    with np.load(source,allow_pickle=False) as maps:
        valid = maps['valid_mask'].astype(bool)
        if not valid.any():
            raise ValueError('No valid supervision pixels for paper export')
        maximum_decrease = float(-maps['delta_w_phy'][valid].min())
    delta_limit = max(.10,float(np.ceil(maximum_decrease/.05)*.05))
    files = render_phy_paper(source, output, weight_range=(0,1),delta_range=(-delta_limit,0))
    print('Paper figure:', files[0])
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only',action='store_true',help='redraw saved NPZ without model inference')
    parser.add_argument('--responses',type=Path,help='explicit saved NPZ for --render-only')
    parser.add_argument('--paper-out',type=Path,help='separate paper figure output directory')
    parser.add_argument('--checkpoint',type=Path,
        help='checkpoint from this physiology experiment, absolute or relative to work_dir')
    args = parser.parse_args(argv)
    if args.responses and not args.render_only:
        parser.error('--responses requires --render-only')
    if args.checkpoint and args.render_only:
        parser.error('--checkpoint requires forward replay')
    if args.render_only and args.responses:
        render_paper_result(args.responses,args.paper_out)
        return 0
    from mmcv import Config
    from mmseg.utils.mechanism import deterministic_probe_config

    config = Config.fromfile(str(CONFIG))
    modules = {name: bool(config.uda[key].get('enable', False)) for name, key in
               (('phy', 'phy_cfg'), ('struct', 'struct_cfg'),
                ('proto', 'proto_cfg'), ('edge', 'struct_edge_cfg'))}
    if modules != MODULES:
        raise ValueError('This entry point requires the physiology-only experiment')
    work_dir = Path(config.work_dir)
    if args.render_only:
        folder = work_dir/'analysis/fixed_probe/replayed/iter_7000_phy_evidence'/Path(FILENAME).stem
        sources = list(folder.glob('iter_007000_*/responses.npz'))
        if len(sources) != 1:
            raise ValueError('Expected one saved Iter 7000 response file; specify --responses explicitly')
        render_paper_result(sources[0],args.paper_out)
        return 0
    checkpoint = (work_dir / (args.checkpoint or 'best_mIoU_iter_7000.pth')).resolve()
    probe = deterministic_probe_config(config)
    if probe.get('filename') != FILENAME:
        raise ValueError('Configuration selected an unexpected target image')
    tif_path = Path(probe['expected_path'])
    target = config.data.train.target
    suffix = target.get('index_suffix', '_dir')
    split = Path(config.data.val.img_dir).name
    annotation = reference_path(config, FILENAME)
    required_files = [checkpoint, tif_path, annotation] + [
        Path(config.data.val.data_root) / (name + suffix) / split / FILENAME
        for name in target.index_names]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError('Required replay input is missing: ' + str(path))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the active Python environment')

    digest = file_digest(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu')
    iteration = checkpoint_iteration(checkpoint, saved.get('meta', {}))
    step = iteration['completed_iter']
    state = checkpoint_state(saved)
    record = dict(path=str(checkpoint), status='readable',
        checkpoint_sha256=digest, state_sha256=state_digest(state), **iteration)
    del state, saved
    output = work_dir / 'analysis/fixed_probe/replayed' / ('phy_{}_{}'.format(
        step,datetime.now().strftime('%m%d_%H%M%S_%f')))
    output.mkdir(parents=True,exist_ok=False)
    print('Replay output:',output)
    write_json(output / 'checkpoint_audit.json', dict(
        data_source='checkpoint_replay', checkpoints=[record]))
    result = replay_node(config, record, output, torch.device('cuda:0'), phy_evidence=True)
    if result['status'] != 'complete':
        raise RuntimeError('Incomplete checkpoint recovery: ' + json.dumps(result['recovery']))
    folder = Path(result['output'])
    verify_output(folder, tif_path, step)
    if file_digest(checkpoint) != digest:
        raise RuntimeError('Checkpoint changed during replay')
    paper_files = render_paper_result(folder/'responses.npz',args.paper_out)
    write_json(output / ('replay_report_{}.json'.format(step)), dict(
        data_source='checkpoint_replay', results=[result], paper_figures=[str(path) for path in paper_files],
        semantic_statistics=str(paper_files[0].parent/'semantic_statistics.json')))
    print('PASS - checkpoint replay, spatial verification and paper export completed')
    print('Figure:', paper_files[0])
    print('Paper PDF:', paper_files[1])
    print('Aligned individual images:', paper_files[0].parent/'spatial')
    print('Cropped RGB TIF:', paper_files[0].parent/'spatial/target_image.tif')
    print('Semantic check map:', paper_files[0].parent/'spatial/phy_semantic_check.png')
    print('Semantic statistics:',paper_files[0].parent/'semantic_statistics.csv')
    print('Semantic interpretation:',paper_files[0].parent/'semantic_summary.txt')
    print('Full-grid figure:', folder / ('iter_{:06d}_phys.png'.format(step)))
    print('Metadata:', folder / 'replay_metadata.json')
    print('Class statistics:', folder / 'phy_class_statistics.csv')
    print('GT reference:', paper_files[0].parent/'spatial/reference_label.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
