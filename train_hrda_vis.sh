#!/bin/bash
# 如果需要激活 conda 环境，请取消下面两行的注释，并替换为你的环境名
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

echo "===== 开始训练任务 1/2：goji30RGB → goji26IRRG (hrda, ndvi) ====="
python tools/train.py configs/hrda/goji30RGB2goji26IRRG_hrda_ndvi.py

echo "===== 开始训练任务 2/2：goji30RGB → goji26IRRG (hrda, 3vi) ====="
python tools/train.py configs/hrda/goji30RGB2goji26IRRG_hrda_3vi.py

echo "✅ 所有 hrda 训练任务已完成"