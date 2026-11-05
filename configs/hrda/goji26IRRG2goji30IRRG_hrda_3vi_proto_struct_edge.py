_base_ = ['./goji26IRRG2goji30IRRG_hrda_3vi_ps_proto.py']

from configs.hrda.sg_bde_proto_edge_common import make_sg_bde_proto_edge

model, uda, log_config = make_sg_bde_proto_edge('struct')
del make_sg_bde_proto_edge

work_dir = '/root/shared-nvme/work_dirs/goji26IRRG2goji30IRRG_hrda_3vi_proto_struct_edge'
name = 'goji26IRRG2goji30IRRG_hrda_3vi_proto_struct_edge'
exp = 'goji_uda_indices_proto_struct_edge'
name_dataset = 'goji26IRRG2goji30IRRG_256x256_3vi_proto_struct_edge'
name_uda = 'dacs_3vi_proto_struct_edge'
name_opt = 'adamw_6e-05_poly10warm_15k'
