# Copyright (c) OpenMMLab. All rights reserved.
# Modifications: Support crop_pseudo_margins

import os.path as osp

import numpy as np
from mmcv.utils import print_log

from .builder import DATASETS
from .custom import CustomDataset
from .goji_meta import GOJI_CLASSES, GOJI_PALETTE
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class GojiDataset(CustomDataset):
    """Goji dataset."""

    CLASSES = GOJI_CLASSES

    PALETTE = GOJI_PALETTE

    def __init__(self,
                 crop_pseudo_margins=None,
                 use_indices=False,
                 use_vi=False,
                 vi_names=None,
                 vi_as_input=True,
                 index_names=None,
                 index_suffix='_dir',
                 vi_norm_cfg=None,
                 vi_norm_for_input=False,
                 **kwargs):
        if crop_pseudo_margins is not None:
            assert kwargs['pipeline'][-1]['type'] == 'Collect'
            kwargs['pipeline'][-1]['keys'].append('valid_pseudo_mask')
        self.use_indices = use_indices
        self.use_vi = use_vi
        self.vi_as_input = vi_as_input
        self.index_names = index_names or vi_names or []
        self.vi_names = vi_names or self.index_names
        self.index_suffix = index_suffix
        self.vi_norm_cfg = vi_norm_cfg
        self.vi_norm_for_input = vi_norm_for_input
        super(GojiDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=True,
            **kwargs)
        self.pseudo_margins = crop_pseudo_margins
        self.valid_mask_size = [256, 256]
        self._log_index_cfg(kwargs['pipeline'])

    def _log_index_cfg(self, pipeline):
        logger = get_root_logger()
        print_log(
            f'Goji dataset use_indices={self.use_indices}', logger=logger)
        print_log(f'Goji dataset use_vi={self.use_vi}', logger=logger)
        print_log(f'Goji dataset vi_as_input={self.vi_as_input}',
                  logger=logger)
        print_log(
            f'Goji dataset index_names={self.index_names}', logger=logger)
        input_channels = 3 + len(self.index_names) \
            if self.use_indices and self.vi_as_input else 3
        print_log(f'Input channels={input_channels}', logger=logger)
        for transform in pipeline:
            if transform.get('type') == 'Normalize':
                print_log(
                    f"img_norm_cfg mean length={len(transform['mean'])}",
                    logger=logger)
                print_log(
                    f"img_norm_cfg std length={len(transform['std'])}",
                    logger=logger)
                break

    def pre_pipeline(self, results):
        super(GojiDataset, self).pre_pipeline(results)
        results['use_indices'] = self.use_indices
        results['use_vi'] = self.use_vi
        results['vi_as_input'] = self.vi_as_input
        results['index_names'] = self.index_names
        results['vi_names'] = self.vi_names
        results['index_suffix'] = self.index_suffix
        results['vi_norm_cfg'] = self.vi_norm_cfg
        results['vi_norm_for_input'] = self.vi_norm_for_input
        if self.use_indices or self.use_vi:
            split = osp.basename(self.img_dir)
            data_root = osp.dirname(osp.dirname(self.img_dir))
            results['index_prefixes'] = {
                name: osp.join(
                    data_root,
                    f'{name}{self.index_suffix}' if self.index_suffix else name,
                    split)
                for name in self.index_names
            }
        if self.pseudo_margins is not None:
            results['valid_pseudo_mask'] = np.ones(
                self.valid_mask_size, dtype=np.uint8)
            if self.pseudo_margins[0] > 0:
                results['valid_pseudo_mask'][:self.pseudo_margins[0], :] = 0
            if self.pseudo_margins[1] > 0:
                results['valid_pseudo_mask'][-self.pseudo_margins[1]:, :] = 0
            if self.pseudo_margins[2] > 0:
                results['valid_pseudo_mask'][:, :self.pseudo_margins[2]] = 0
            if self.pseudo_margins[3] > 0:
                results['valid_pseudo_mask'][:, -self.pseudo_margins[3]:] = 0
            results['seg_fields'].append('valid_pseudo_mask')
