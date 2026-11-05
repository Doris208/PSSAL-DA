#!/bin/bash
set -e

echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (Ours HRDA ResNet101) ====="
python tools/train.py configs/hrda/resnet_101/goji30IRRG2goji26IRRG_ours_resnet101.py

echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (Ours HRDA ResNet101) ====="
python tools/train.py configs/hrda/resnet_101/goji26IRRG2goji30RGB_ours_resnet101.py

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (Ours HRDA ResNet101) ====="
python tools/train.py configs/hrda/resnet_101/goji26RGB2goji30RGB_ours_resnet101.py

echo "All Ours HRDA ResNet101 training tasks finished."
