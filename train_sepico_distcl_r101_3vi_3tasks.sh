#!/usr/bin/env bash
set -e

python tools/train.py configs/sepico/goji30IRRG2goji26IRRG_sepico_distcl_r101_3vi.py 

python tools/train.py configs/sepico/goji26IRRG2goji30RGB_sepico_distcl_r101_3vi.py 

python tools/train.py configs/sepico/goji26RGB2goji30RGB_sepico_distcl_r101_3vi.py 

echo "All Sepico-DistCL-R101 + 3VI training tasks finished."

echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (ProCA-R101 + 3VI) ====="
python tools/train.py configs/proca/goji30IRRG2goji26IRRG_proca_r101_3vi.py

echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (ProCA-R101 + 3VI) ====="
python tools/train.py configs/proca/goji26IRRG2goji30RGB_proca_r101_3vi.py

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (ProCA-R101 + 3VI) ====="
python tools/train.py configs/proca/goji26RGB2goji30RGB_proca_r101_3vi.py

echo "All ProCA-R101 + 3VI training tasks finished."