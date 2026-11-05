#!/bin/bash
# Uncomment and edit these lines if you need to activate a conda env.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

echo "===== Start training task 1/6: goji26IRRG -> goji30IRRG (hrda, 3vi_proto_full_edge) ====="
python tools/train.py configs/hrda/goji26IRRG2goji30IRRG_hrda_3vi_proto_full_edge.py

echo "===== Start training task 2/6: goji30IRRG -> goji26IRRG (hrda, 3vi_proto_full_edge) ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_proto_full_edge.py

echo "===== Start training task 3/6: goji26IRRG -> goji30RGB (hrda, 3vi_proto_full_edge) ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_full_edge.py

# echo "===== Start training task 4/6: goji30RGB -> goji26IRRG (hrda, 3vi_proto_full_edge) ====="
# python tools/train.py configs/hrda/goji30RGB2goji26IRRG_hrda_3vi_proto_full_edge.py

echo "===== Start training task 5/6: goji26RGB -> goji30RGB (hrda, 3vi_proto_full_edge) ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_proto_full_edge.py

# echo "===== Start training task 6/6: goji30RGB -> goji26RGB (hrda, 3vi_proto_full_edge) ====="
# python tools/train.py configs/hrda/goji30RGB2goji26RGB_hrda_3vi_proto_full_edge.py

echo "All hrda 3vi_proto_full_edge training tasks finished."