#!/bin/bash
set -e
# Uncomment and edit these lines if you need to activate a conda env.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

echo "===== Start training: goji26RGB -> goji30RGB (hrda, 3vi_struct_edge) ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_struct_edge.py

echo "hrda 3vi_struct_edge training finished."
