#!/bin/bash
set -e

echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (DAFormer 3VI) ====="
python tools/train.py configs/daformer/goji30IRRG2goji26IRRG_daformer_3vi.py
  
echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (DAFormer 3VI) ====="
python tools/train.py configs/daformer/goji26IRRG2goji30RGB_daformer_3vi.py 

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (DAFormer 3VI) ====="
python tools/train.py configs/daformer/goji26RGB2goji30RGB_daformer_3vi.py 

echo "All DAFormer 3VI training tasks finished."
