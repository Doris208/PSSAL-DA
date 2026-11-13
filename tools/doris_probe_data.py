"""NumPy-only validation shared by online capture and saved-array rendering."""
import numpy as np


RGB30_DIRECTIONS = ('goji26RGB2goji30RGB', 'goji26IRRG2goji30RGB')


def replay_direction(config):
    """Resolve supported source views from data identity and verify experiment identity."""
    from pathlib import Path
    source = config['data']['train']['source']
    target = config['data']['train']['target']
    source_views = {'rgb_dir/train': 'RGB', 'img_dir/train': 'IRRG'}
    view = source_views.get(Path(source['img_dir']).as_posix())
    if (Path(source['data_root']).name != '26' or view is None or
            Path(target['data_root']).name != '30' or
            Path(target['img_dir']).as_posix() != 'rgb_dir/train'):
        raise ValueError('Expected source RGB26/IRRG26 and target RGB30 dataset identity')
    direction = 'goji26' + view + '2goji30RGB'
    if (not config.get('name', '').startswith('doris_' + direction + '_hrda_3vi') or
            config.get('name_dataset') != direction + '_256x256_3vi'):
        raise ValueError('Experiment direction differs from source/target dataset identity')
    return direction


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
