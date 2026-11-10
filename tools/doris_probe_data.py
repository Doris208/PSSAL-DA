"""NumPy-only validation shared by online capture and saved-array rendering."""
import numpy as np


def spatial_statistics(maps):
    """Describe real full-grid responses. Reject unexplained spatial transforms."""
    shape = maps['valid_mask'].shape
    valid = maps['valid_mask'].astype(bool)
    stats = {}
    for key, value in maps.items():
        value = np.asarray(value)
        if value.ndim != 2:
            continue
        if value.shape != shape:
            raise ValueError('Unmapped spatial grid for {}: {} != {}'.format(key, value.shape, shape))
        if not np.isfinite(value).all():
            raise ValueError('Non-finite mechanism response: ' + key)
        values = value[valid]
        stats[key] = dict(shape=list(shape), valid_count=int(values.size),
            min=float(values.min()) if values.size else None,
            max=float(values.max()) if values.size else None,
            mean=float(values.mean()) if values.size else None,
            std=float(values.std()) if values.size else None,
            nonzero_ratio=float(np.count_nonzero(values)/values.size) if values.size else None)
    return stats
