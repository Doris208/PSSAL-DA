#!/bin/bash
# Uncomment and edit these lines if you need to activate a conda env.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

echo "===== Start training task 1/7: 26IRRG (pssal, 00_supervised_3vi_control) ====="
python tools/train.py configs/hrda/pssal/goji26IRRG_00_supervised_3vi_control.py

echo "===== Start training task 2/7: 30RGB (pssal, 00_supervised_3vi_control) ====="
python tools/train.py configs/hrda/pssal/goji30RGB_00_supervised_3vi_control.py

echo "===== Start training task 3/7: 26RGB (pssal, 00_supervised_3vi_control) ====="
python tools/train.py configs/hrda/pssal/goji26RGB_00_supervised_3vi_control.py

echo "===== Start training task 4/7: 30IRRG (pssal, 00_supervised_3vi_control) ====="
python tools/train.py configs/hrda/pssal/goji30IRRG_00_supervised_3vi_control.py

echo "===== Start training task 5/7: 30IRRG2goji26IRRG (pssal, 00_supervised_3vi_control) ====="
python tools/train.py configs/hrda/pssal/goji30IRRG2goji26IRRG_00_supervised_3vi_control.py

echo "===== Start training task 6/7: 26IRRG2goji30RGB (pssal, 00_supervised_3vi_control) ====="
python tools/train.py configs/hrda/pssal/goji26IRRG2goji30RGB_00_supervised_3vi_control.py

echo "===== Start training task 7/7: 26RGB2goji30RGB (pssal, 00_supervised_3vi_control) ====="
python tools/train.py configs/hrda/pssal/goji26RGB2goji30RGB_00_supervised_3vi_control.py

echo "All pssal 00_supervised_3vi_control training tasks finished."
