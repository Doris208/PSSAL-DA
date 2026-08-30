_base_ = ['./goji26RGB2goji30RGB_hrda_3vi_ps_proto.py']

from configs.hrda.sg_bde_proto_edge_common import make_sg_bde_proto_edge

model, uda, log_config = make_sg_bde_proto_edge('full')
del make_sg_bde_proto_edge

work_dir = 'shared-nvme/work_dirs/goji26RGB2goji30RGB_hrda_3vi_proto_full_edge'
name = 'goji26RGB2goji30RGB_hrda_3vi_proto_full_edge'
exp = 'goji_uda_indices_proto_full_edge'
name_dataset = 'goji26RGB2goji30RGB_256x256_3vi_proto_full_edge'
name_uda = 'dacs_3vi_proto_full_edge'
name_opt = 'adamw_6e-05_poly10warm_15k'
