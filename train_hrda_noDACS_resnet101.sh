#!/bin/bash
set -e

echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (HRDA noDACS ResNet101 + 3VI) ====="
python tools/train.py configs/hrda/noDACS_resnet/goji30IRRG2goji26IRRG_noDACS_resnet101.py

echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (HRDA noDACS ResNet101 + 3VI) ====="
python tools/train.py configs/hrda/noDACS_resnet/goji26IRRG2goji30RGB_noDACS_resnet101.py

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (HRDA noDACS ResNet101 + 3VI) ====="
python tools/train.py configs/hrda/noDACS_resnet/goji26RGB2goji30RGB_noDACS_resnet101.py

echo "All HRDA noDACS ResNet101 + 3VI training tasks finished."
