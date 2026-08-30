import sys

from mmcv import Config


CONFIGS = [
    ('Baseline', 'configs/hrda/goji26RGB2goji30RGB_hrda_3vi.py',
     (0, 0, 0, 0)),
    ('Physiology', 'configs/hrda/goji26RGB2goji30RGB_hrda_3vi_phy_gate.py',
     (1, 0, 0, 0)),
    ('Structural', 'configs/hrda/goji26RGB2goji30RGB_hrda_3vi_struct_gate.py',
     (0, 1, 0, 0)),
    ('MPARC', 'configs/hrda/goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py',
     (1, 1, 0, 0)),
    ('Prototype', 'configs/hrda/goji26RGB2goji30RGB_hrda_3vi_ps_proto.py',
     (0, 0, 1, 0)),
    ('Edge', 'configs/hrda/goji26RGB2goji30RGB_hrda_3vi_struct_edge.py',
     (0, 0, 0, 1)),
    ('PBCR', 'configs/hrda/goji26RGB2goji30RGB_hrda_3vi_pbcr.py',
     (0, 0, 1, 1)),
    ('MPARC+PBCR',
     'configs/hrda/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py',
     (1, 1, 1, 1)),
]


def flag(cfg, key):
    return int(cfg.uda.get(key, {}).get('enable', False))


def main():
    work_dirs = set()
    ok = True
    for name, path, expected in CONFIGS:
        cfg = Config.fromfile(path)
        actual = (
            flag(cfg, 'phy_cfg'),
            flag(cfg, 'struct_cfg'),
            flag(cfg, 'proto_cfg'),
            flag(cfg, 'struct_edge_cfg'),
        )
        source = cfg.data.train.source
        target = cfg.data.train.target
        checks = [
            (actual == expected, f'flags expected {expected}, got {actual}'),
            (source.data_root.endswith('/26'), 'source root is not /26'),
            (target.data_root.endswith('/30'), 'target root is not /30'),
            (source.img_dir == 'rgb_dir/train', 'source is not RGB train'),
            (target.img_dir == 'rgb_dir/train', 'target is not RGB train'),
            (cfg.model.backbone.in_chans == 6, 'backbone is not 6-channel'),
            (cfg.model.decode_head.num_classes == 5,
             'decode head is not 5 classes'),
            (cfg.runner.max_iters == 15000, 'max_iters is not 15000'),
            (cfg.checkpoint_config.interval == 1000,
             'checkpoint interval is not 1000'),
            (cfg.evaluation.interval == 1000,
             'evaluation interval is not 1000'),
            (cfg.work_dir not in work_dirs, 'work_dir is duplicated'),
        ]
        work_dirs.add(cfg.work_dir)
        for passed, message in checks:
            if not passed:
                ok = False
                print(f'[FAIL] {name}: {message}')
        print(f'[OK] {name}: flags={actual}, work_dir={cfg.work_dir}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
