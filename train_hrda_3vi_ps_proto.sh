#!/bin/bash
# 如果需要激活 conda 环境，请取消下面两行的注释，并替换为你的环境名
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

echo "===== 开始训练任务 1/6：goji26IRRG → goji30IRRG (hrda, 3vi_ps_proto) ====="
python tools/train.py configs/hrda/goji26IRRG2goji30IRRG_hrda_3vi_ps_proto.py 

echo "===== 开始训练任务 2/6：goji30IRRG → goji26IRRG (hrda, 3vi_ps_proto) ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_ps_proto.py

echo "===== 开始训练任务 3/6：goji26IRRG → goji30RGB (hrda, 3vi_ps_proto) ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_ps_proto.py

echo "===== 开始训练任务 4/6：goji30RGB → goji26IRRG (hrda, 3vi_ps_proto) ====="
python tools/train.py configs/hrda/goji30RGB2goji26IRRG_hrda_3vi_ps_proto.py

echo "===== 开始训练任务 5/6：goji26RGB → goji30RGB (hrda, 3vi_ps_proto) ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_ps_proto.py

echo "===== 开始训练任务 6/6：goji30RGB → goji26RGB (hrda, 3vi_ps_proto) ====="
python tools/train.py configs/hrda/goji30RGB2goji26RGB_hrda_3vi_ps_proto.py

echo "✅ 所有 hrda 训练任务已完成"