_base_ = ['./goji26IRRG2goji30RGB_hrda_3vi_ps_proto.py']

from configs.hrda.pssalda_paper_pbcr_common import make_paper_pbcr_edge

model, uda, log_config = make_paper_pbcr_edge(enable_proto=True)
del make_paper_pbcr_edge

work_dir = 'shared-nvme/work_dirs/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge_new'
name = 'goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge_new'
exp = 'goji_uda_indices_proto_struct_edge_new'
name_dataset = 'goji26IRRG2goji30RGB_256x256_3vi_proto_struct_edge_new'
name_uda = 'dacs_3vi_proto_struct_edge_new'
name_opt = 'adamw_6e-05_poly10warm_15k'
