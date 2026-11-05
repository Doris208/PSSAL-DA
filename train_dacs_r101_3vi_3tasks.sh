#!/bin/bash
set -e

echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (DACS-R101 + 3VI) ====="
python tools/train.py configs/dacs/goji30IRRG2goji26IRRG_dacs_r101_3vi.py 

echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (DACS-R101 + 3VI) ====="
python tools/train.py configs/dacs/goji26IRRG2goji30RGB_dacs_r101_3vi.py 

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (DACS-R101 + 3VI) ====="
python tools/train.py configs/dacs/goji26RGB2goji30RGB_dacs_r101_3vi.py 

echo "All DACS-R101 + 3VI training tasks finished."
