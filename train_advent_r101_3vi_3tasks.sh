#!/bin/bash
set -e

echo "===== Start training task 1/3: goji30IRRG -> goji26IRRG (AdvEnt-R101 + 3VI) ====="
python tools/train.py configs/advent/goji30IRRG2goji26IRRG_advent_r101_3vi.py

echo "===== Start training task 2/3: goji26IRRG -> goji30RGB (AdvEnt-R101 + 3VI) ====="
python tools/train.py configs/advent/goji26IRRG2goji30RGB_advent_r101_3vi.py

echo "===== Start training task 3/3: goji26RGB -> goji30RGB (AdvEnt-R101 + 3VI) ====="
python tools/train.py configs/advent/goji26RGB2goji30RGB_advent_r101_3vi.py

echo "All AdvEnt-R101 + 3VI training tasks finished."
