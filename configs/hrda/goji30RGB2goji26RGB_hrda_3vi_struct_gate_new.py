_base_ = ['./goji30RGB2goji26RGB_hrda_3vi_struct_gate.py']

from configs.hrda.pssalda_paper_pbcr_common import make_feature_interaction

model, uda, log_config = make_feature_interaction()
del make_feature_interaction

work_dir = '/root/shared-nvme/work_dirs/goji30RGB2goji26RGB_hrda_3vi_struct_gate_new'
name = 'goji30RGB2goji26RGB_hrda_3vi_struct_gate_new'
exp = 'goji_uda_indices_struct_gate_new'
name_dataset = 'goji30RGB2goji26RGB_256x256_3vi_struct_gate_new'
name_uda = 'dacs_3vi_struct_gate_new'
name_opt = 'adamw_6e-05_poly10warm_15k'
