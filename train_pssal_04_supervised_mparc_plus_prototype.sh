#!/bin/bash
# Uncomment and edit these lines if you need to activate a conda env.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

echo "===== Start training task 1/7: 26IRRG (pssal, 04_supervised_mparc_plus_prototype) ====="
python tools/train.py configs/hrda/pssal/goji26IRRG_04_supervised_mparc_plus_prototype.py

echo "===== Start training task 2/7: 30RGB (pssal, 04_supervised_mparc_plus_prototype) ====="
python tools/train.py configs/hrda/pssal/goji30RGB_04_supervised_mparc_plus_prototype.py

echo "===== Start training task 3/7: 26RGB (pssal, 04_supervised_mparc_plus_prototype) ====="
python tools/train.py configs/hrda/pssal/goji26RGB_04_supervised_mparc_plus_prototype.py

echo "===== Start training task 4/7: 30IRRG (pssal, 04_supervised_mparc_plus_prototype) ====="
python tools/train.py configs/hrda/pssal/goji30IRRG_04_supervised_mparc_plus_prototype.py

echo "===== Start training task 5/7: 30IRRG2goji26IRRG (pssal, 04_supervised_mparc_plus_prototype) ====="
python tools/train.py configs/hrda/pssal/goji30IRRG2goji26IRRG_04_supervised_mparc_plus_prototype.py

echo "===== Start training task 6/7: 26IRRG2goji30RGB (pssal, 04_supervised_mparc_plus_prototype) ====="
python tools/train.py configs/hrda/pssal/goji26IRRG2goji30RGB_04_supervised_mparc_plus_prototype.py

echo "===== Start training task 7/7: 26RGB2goji30RGB (pssal, 04_supervised_mparc_plus_prototype) ====="
python tools/train.py configs/hrda/pssal/goji26RGB2goji30RGB_04_supervised_mparc_plus_prototype.py

echo "All pssal 04_supervised_mparc_plus_prototype training tasks finished."
