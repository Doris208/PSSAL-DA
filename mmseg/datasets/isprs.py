# Copyright (c) OpenMMLab. All rights reserved.
# Modifications: Support crop_pseudo_margins

import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ISPRSDataset(CustomDataset):
    """ISPRS dataset."""

    CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')

    PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0],
               [0, 255, 255], [0, 0, 255]]

    def __init__(self, crop_pseudo_margins=None, **kwargs):
        if crop_pseudo_margins is not None:
            assert kwargs['pipeline'][-1]['type'] == 'Collect'
            kwargs['pipeline'][-1]['keys'].append('valid_pseudo_mask')
        super(ISPRSDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
        self.pseudo_margins = crop_pseudo_margins
        # ISPRS samples are resized to 1024x1024 before RandomCrop.
        self.valid_mask_size = [1024, 1024]

    def pre_pipeline(self, results):
        super(ISPRSDataset, self).pre_pipeline(results)
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
