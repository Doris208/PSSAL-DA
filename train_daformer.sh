#!/bin/bash
# 如果需要激活 conda 环境，请取消下面两行的注释，并替换为你的环境名
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

echo "===== 开始训练任务 1/5：goji30IRRG → goji26IRRG ====="
python tools/train.py configs/daformer/goji30IRRG2goji26IRRG_daformer.py

echo "===== 开始训练任务 2/5：goji26IRRG → goji30RGB ====="
python tools/train.py configs/daformer/goji26IRRG2goji30RGB_daformer.py

echo "===== 开始训练任务 3/5：goji30RGB → goji26IRRG ====="
python tools/train.py configs/daformer/goji30RGB2goji26IRRG_daformer.py

echo "===== 开始训练任务 4/5：goji26RGB → goji30RGB ====="
python tools/train.py configs/daformer/goji26RGB2goji30RGB_daformer.py

echo "===== 开始训练任务 5/5：goji30RGB → goji26RGB ====="
python tools/train.py configs/daformer/goji30RGB2goji26RGB_daformer.py

echo "✅ 所有训练任务已完成"