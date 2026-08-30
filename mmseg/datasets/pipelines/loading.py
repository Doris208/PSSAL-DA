# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp
from io import BytesIO

import mmcv
import numpy as np
from PIL import Image

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadGojiIndices(object):
    """Load Goji vegetation index maps.

    The default behavior keeps backward compatibility and concatenates raw
    indices to ``img`` when ``use_indices=True`` and ``vi_as_input=True``.
    When ``use_vi=True``, normalized indices are also returned as ``vi_img``.
    """

    def __init__(self, file_client_args=dict(backend='disk')):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self._logged_stats = False

    def _load_index(self, filename):
        img_bytes = self.file_client.get(filename)
        index = np.array(Image.open(BytesIO(img_bytes)))
        if index.ndim == 3:
            index = index[..., 0]
        index = index.astype(np.float32)
        return np.nan_to_num(index, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_index(self, index, index_name, norm_cfg):
        if norm_cfg is None:
            norm_cfg = {}
        cfg = norm_cfg.get(index_name, norm_cfg)
        min_val = cfg.get('min', -1.0)
        max_val = cfg.get('max', 1.0)
        index = np.clip(index, min_val, max_val)
        index = (index - min_val) / max(max_val - min_val, 1e-6)
        return index.astype(np.float32)

    def __call__(self, results):
        if not results.get('use_indices', False) and \
                not results.get('use_vi', False):
            return results

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        img = results['img']
        if img.ndim == 2:
            img = img[..., None]

        input_index_channels = []
        vi_index_channels = []
        names = results.get('vi_names') or results.get('index_names', [])
        for index_name in names:
            filename = osp.join(results['index_prefixes'][index_name],
                                results['img_info']['filename'])
            index = self._load_index(filename)
            if index.shape[:2] != img.shape[:2]:
                raise ValueError(
                    f'Goji index shape mismatch for {filename}: '
                    f'img={img.shape[:2]}, {index_name}={index.shape[:2]}')
            if results.get('use_indices', False) and \
                    results.get('vi_as_input', True):
                input_index = index
                if results.get('vi_norm_for_input', False):
                    input_index = self._normalize_index(
                        input_index, index_name, results.get('vi_norm_cfg'))
                input_index_channels.append(input_index[..., None])
            if results.get('use_vi', False):
                vi_index = self._normalize_index(
                    index, index_name, results.get('vi_norm_cfg'))
                vi_index_channels.append(vi_index[..., None])

        if vi_index_channels:
            vi_img = np.concatenate(vi_index_channels, axis=2)
            results['vi_img'] = vi_img
            results.setdefault('vi_fields', []).append('vi_img')
            if not self._logged_stats:
                stats = []
                for i, name in enumerate(names):
                    stats.append(
                        f'{name}: min={vi_img[..., i].min():.4f}, '
                        f'max={vi_img[..., i].max():.4f}, '
                        f'mean={vi_img[..., i].mean():.4f}, '
                        f'std={vi_img[..., i].std():.4f}')
                mmcv.print_log('Goji VI stats after normalization - ' +
                               '; '.join(stats), 'mmseg')
                self._logged_stats = True

        if input_index_channels:
            results['img'] = np.concatenate(
                [img.astype(np.float32)] + input_index_channels, axis=2)
            results['img_shape'] = results['img'].shape
            results['ori_shape'] = results['img'].shape
            results['pad_shape'] = results['img'].shape
            results['num_channels'] = results['img'].shape[2]

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'
