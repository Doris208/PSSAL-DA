_base_ = ['./goji30RGB2goji26RGB_hrda_3vi_ps_proto.py']

from configs.hrda.sg_bde_proto_edge_common import make_sg_bde_proto_edge

model, uda, log_config = make_sg_bde_proto_edge('spec')
del make_sg_bde_proto_edge

work_dir = '/root/shared-nvme/work_dirs/goji30RGB2goji26RGB_hrda_3vi_proto_spec_edge'
name = 'goji30RGB2goji26RGB_hrda_3vi_proto_spec_edge'
exp = 'goji_uda_indices_proto_spec_edge'
name_dataset = 'goji30RGB2goji26RGB_256x256_3vi_proto_spec_edge'
name_uda = 'dacs_3vi_proto_spec_edge'
name_opt = 'adamw_6e-05_poly10warm_15k'
