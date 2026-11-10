"""VI preprocessing shared by training and source-only evidence analysis."""
import numpy as np


def prepare_goji_index(index):
    if index.ndim == 3:
        index = index[..., 0]
    return np.nan_to_num(index.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_goji_index(index, index_name, norm_cfg=None):
    norm_cfg = norm_cfg or {}
    cfg = norm_cfg.get(index_name, norm_cfg)
    minimum, maximum = cfg.get('min', -1.0), cfg.get('max', 1.0)
    index = np.clip(index, minimum, maximum)
    return ((index - minimum) / max(maximum - minimum, 1e-6)).astype(np.float32)
