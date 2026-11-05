#!/bin/bash
set -e
# Uncomment and edit these lines if you need to activate a conda env.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

echo "===== Start training: goji26RGB -> goji30RGB (hrda, 3vi_pbcr) ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_pbcr.py

echo "===== Start training task 5/6: goji26RGB -> goji30RGB (hrda, 3vi_phy_struct_gate_new) ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate_new.py

echo "===== Start training task 5/6: goji26RGB -> goji30RGB (hrda, 3vi_proto_struct_edge_new) ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge_new.py

echo "hrda 3vi_pbcr training finished."
