_base_ = ['./goji26IRRG2goji30IRRG_hrda_3vi_ps_proto.py']

from configs.hrda.pssalda_paper_pbcr_common import make_feature_interaction

model, uda, log_config = make_feature_interaction()
del make_feature_interaction

work_dir = 'shared-nvme/work_dirs/goji26IRRG2goji30IRRG_hrda_3vi_ps_proto_new'
name = 'goji26IRRG2goji30IRRG_hrda_3vi_ps_proto_new'
exp = 'goji_uda_indices_ps_proto_new'
name_dataset = 'goji26IRRG2goji30IRRG_256x256_3vi_ps_proto_new'
name_uda = 'dacs_3vi_ps_proto_new'
name_opt = 'adamw_6e-05_poly10warm_15k'
