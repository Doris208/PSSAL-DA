#!/bin/bash
set -e

echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (ProCA-R101 + 3VI) ====="
python tools/train.py configs/proca/goji30IRRG2goji26IRRG_proca_r101_3vi.py

echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (ProCA-R101 + 3VI) ====="
python tools/train.py configs/proca/goji26IRRG2goji30RGB_proca_r101_3vi.py

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (ProCA-R101 + 3VI) ====="
python tools/train.py configs/proca/goji26RGB2goji30RGB_proca_r101_3vi.py

echo "All ProCA-R101 + 3VI training tasks finished."
