# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support debug_output_attention

import os.path as osp
import tempfile

import mmcv
import numpy as np
import torch
from matplotlib import cm
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from PIL import Image


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def save_color_seg(seg, palette, out_file):
    """Save prediction as an opaque 3-channel RGB PNG using dataset palette."""
    palette = np.asarray(palette, dtype=np.uint8)
    assert palette.ndim == 2 and palette.shape[1] == 3

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label] = color

    mmcv.mkdir_or_exist(osp.dirname(out_file))
    Image.fromarray(color_seg, mode='RGB').save(out_file)


def save_confidence_map(confidence_map, out_file):
    """Save confidence map derived from max softmax probability."""
    conf_norm = np.clip(confidence_map, 0.0, 1.0)
    conf_color = (cm.get_cmap('jet')(conf_norm)[..., :3] * 255).astype(
        np.uint8)
    mmcv.mkdir_or_exist(osp.dirname(out_file))
    Image.fromarray(conf_color, mode='RGB').save(out_file)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    save_confidence=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        save_confidence (bool): Explicitly export confidence maps, requiring
            out_dir and an additional inference pass. Default: False.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            seg_prob = None
            if out_dir is not None and save_confidence:
                module = model.module if hasattr(model, 'module') else model
                decode_head = getattr(module, 'decode_head', None)
                if not (hasattr(decode_head, 'debug_output_attention')
                        and decode_head.debug_output_attention):
                    img = data['img'][0].to(
                        next(module.parameters()).device, non_blocking=True)
                    img_metas = data['img_metas'][0].data[0]
                    seg_prob = module.inference(img, img_metas, rescale=True)

        if show or out_dir:
            img_metas = data['img_metas'][0].data[0]
            if show:
                img_tensor = data['img'][0]
                img_norm_cfg = img_metas[0]['img_norm_cfg']
                vis_img_tensor = img_tensor[:, :3, ...]
                imgs = tensor2imgs(
                    vis_img_tensor,
                    mean=img_norm_cfg['mean'][:3],
                    std=img_norm_cfg['std'][:3],
                    to_rgb=img_norm_cfg.get('to_rgb', False))
            else:
                imgs = [None] * len(img_metas)
            assert len(imgs) == len(img_metas)

            for batch_idx, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                if show:
                    vis_channel_order = img_meta['img_norm_cfg'].get(
                        'vis_channel_order', None)
                    if vis_channel_order is not None:
                        img = img[..., vis_channel_order]
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]
                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    target_prediction_file = osp.join(out_dir,
                                                      'target_prediction',
                                                      img_meta['ori_filename'])
                    target_confidence_file = osp.join(
                        out_dir, 'target_confidence_map',
                        img_meta['ori_filename'])
                else:
                    target_prediction_file = None
                    target_confidence_file = None

                module = model.module if hasattr(model, 'module') else model
                decode_head = getattr(module, 'decode_head', None)
                if hasattr(decode_head, 'debug_output_attention') and \
                        decode_head.debug_output_attention:
                    # Attention debug output
                    mmcv.imwrite(result[0] * 255, target_prediction_file)
                else:
                    if target_prediction_file is not None:
                        save_color_seg(result[0], dataset.PALETTE,
                                       target_prediction_file)
                        if seg_prob is not None:
                            confidence_map = torch.max(
                                seg_prob[batch_idx], dim=0)[0].cpu().numpy()
                            save_confidence_map(confidence_map,
                                                target_confidence_file)
                    if show:
                        module.show_result(
                            img_show,
                            result,
                            palette=dataset.PALETTE,
                            show=show,
                            out_file=None,
                            opacity=opacity)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
