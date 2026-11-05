#!/bin/bash
set -e

# This script only benchmarks the two cross-backbone entries requested here:
# 1) HRDA with ResNetV1c-101
# 2) Ours / PSSAL-DA with MiT-B5
#
# Checkpoints are optional for Params/GFLOPs/FPS because these metrics depend
# on the architecture and input shape. Fill them only if you want the raw CSV
# to show Checkpoint Loaded=True.

export HRDA_R101_CKPT_TASK1="/root/shared-nvme/work_dirs/baseline_resnet/goji30IRRG2goji26IRRG_baseline_resnet101/best_mIoU_iter_10000.pth"
export HRDA_R101_CKPT_TASK2="/root/shared-nvme/work_dirs/baseline_resnet/goji26IRRG2goji30RGB_baseline_resnet101/best_mIoU_iter_8000.pth"
export HRDA_R101_CKPT_TASK3="/root/shared-nvme/work_dirs/baseline_resnet/goji26RGB2goji30RGB_baseline_resnet101/best_mIoU_iter_12000.pth"

export OURS_MITB5_CKPT_TASK1="/root/shared-nvme/work_dirs/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge/best_mIoU_iter_12000.pth"
export OURS_MITB5_CKPT_TASK2="/root/shared-nvme/work_dirs/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge/best_mIoU_iter_13000.pth"
export OURS_MITB5_CKPT_TASK3="/root/shared-nvme/work_dirs/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge/best_mIoU_iter_8000.pth"

OUTPUT_DIR=/root/shared-nvme/benchmark_results/hrda_ours_cross_backbone

python tools/analysis/benchmark_external_methods.py \
  --manifest benchmark_hrda_ours_cross_backbone.json \
  --device cuda:0 \
  --output-dir "${OUTPUT_DIR}"

python tools/analysis/aggregate_efficiency_results.py \
  --input-dir "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}/summary"
