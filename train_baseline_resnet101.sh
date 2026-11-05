#!/bin/bash
set -e

echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (HRDA baseline ResNet101 + 3VI) ====="
python tools/train.py configs/hrda/baseline_resnet/goji30IRRG2goji26IRRG_baseline_resnet101.py

echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (HRDA baseline ResNet101 + 3VI) ====="
python tools/train.py configs/hrda/baseline_resnet/goji26IRRG2goji30RGB_baseline_resnet101.py

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (HRDA baseline ResNet101 + 3VI) ====="
python tools/train.py configs/hrda/baseline_resnet/goji26RGB2goji30RGB_baseline_resnet101.py

echo "All HRDA baseline ResNet101 + 3VI training tasks finished."
