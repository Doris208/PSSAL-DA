_base_ = ['./goji30RGB2goji26IRRG_hrda_3vi_ps_proto.py']

from configs.hrda.sg_bde_proto_edge_common import make_sg_bde_proto_edge

model, uda, log_config = make_sg_bde_proto_edge('plain')
del make_sg_bde_proto_edge

work_dir = 'shared-nvme/work_dirs/goji30RGB2goji26IRRG_hrda_3vi_proto_plain_edge'
name = 'goji30RGB2goji26IRRG_hrda_3vi_proto_plain_edge'
exp = 'goji_uda_indices_proto_plain_edge'
name_dataset = 'goji30RGB2goji26IRRG_256x256_3vi_proto_plain_edge'
name_uda = 'dacs_3vi_proto_plain_edge'
name_opt = 'adamw_6e-05_poly10warm_15k'
