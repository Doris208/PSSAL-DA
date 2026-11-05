#!/bin/bash
set -e

echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (MIC HRDA) ====="
python tools/train.py configs/mic/goji30IRRG2goji26IRRG_mic_hrda.py 

echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (MIC HRDA) ====="
python tools/train.py configs/mic/goji26IRRG2goji30RGB_mic_hrda.py 

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (MIC HRDA) ====="
python tools/train.py configs/mic/goji26RGB2goji30RGB_mic_hrda.py 

echo "All MIC HRDA training tasks finished."

echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (Ours HRDA ResNet101) ====="
python tools/train.py configs/hrda/resnet_101/goji30IRRG2goji26IRRG_ours_resnet101.py

echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (Ours HRDA ResNet101) ====="
python tools/train.py configs/hrda/resnet_101/goji26IRRG2goji30RGB_ours_resnet101.py

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (Ours HRDA ResNet101) ====="
python tools/train.py configs/hrda/resnet_101/goji26RGB2goji30RGB_ours_resnet101.py

echo "All Ours HRDA ResNet101 training tasks finished."


echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (MemoryAdaptNet ResNet101 + 3VI) ====="
python tools/train.py configs/memoryadaptnet/goji30IRRG2goji26IRRG_memoryadaptnet_r101_3vi.py

echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (MemoryAdaptNet ResNet101 + 3VI) ====="
python tools/train.py configs/memoryadaptnet/goji26IRRG2goji30RGB_memoryadaptnet_r101_3vi.py

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (MemoryAdaptNet ResNet101 + 3VI) ====="
python tools/train.py configs/memoryadaptnet/goji26RGB2goji30RGB_memoryadaptnet_r101_3vi.py

echo "All MemoryAdaptNet ResNet101 + 3VI training tasks finished."

