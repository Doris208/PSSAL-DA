_base_ = ['./goji26IRRG2goji30RGB_hrda_3vi.py']

uda = dict(
    phy_cfg=dict(enable=False),
    struct_cfg=dict(
        enable=True,
        struct_positive_classes=[2, 3, 4],
        lambda_struct_tgt=0.08,
        use_target_gate=True,
        struct_min_weight=0.60,
        struct_filter_threshold=0.35,
        struct_filter_mode='soft',
        struct_warmup_iters=6000,
        struct_rampup_iters=4000,
        source_adj_momentum=0.99,
        semantic_proto_momentum=0.99,
        use_source_adjacency_ema=True,
        use_semantic_proto=True,
        kernel_size=3,
        confidence_power=1.0,
        continuity_power=1.0,
        adjacency_power=1.0,
        semantic_power=0.5,
        detach_gate=True,
        debug=True,
        save_struct_debug=True,
        struct_debug_interval=1000,
        struct_debug_max_samples=50,
        struct_debug_dir='struct_gate_debug'))

runner = dict(type='IterBasedRunner', max_iters=20000)
work_dir = '/root/shared-nvme/work_dirs/goji26IRRG2goji30RGB_hrda_3vi_struct_gate'
name = 'goji26IRRG2goji30RGB_hrda_3vi_struct_gate'
exp = 'goji_uda_indices_struct'
name_dataset = 'goji26IRRG2goji30RGB_256x256_3vi_struct_gate'
name_uda = 'dacs_3vi_struct_gate'
name_opt = 'adamw_6e-05_poly10warm_20k'
# PSSAL-DA aligned 20k ablation override.
_base_ = ['./pssalda_26irrg2_30rgb_ablation_common.py']
from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(True, False)
uda = make_uda_override(False, True, False, False)
log_config = make_log_config(False, False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
work_dir = '/root/shared-nvme/work_dirs/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_struct_gate'
name = 'goji26IRRG2goji30RGB_hrda_3vi_struct_gate'
exp = 'goji_uda_indices_ablation20k_structural'
name_uda = 'dacs_3vi_structural'
name_opt = 'adamw_6e-05_poly10warm_20k'
del make_log_config, make_model_override, make_uda_override