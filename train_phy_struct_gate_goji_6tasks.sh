#!/bin/bash

echo "===== Start task 1/6: goji26IRRG -> goji30IRRG (hrda, 3vi_phy_struct_gate) ====="
python tools/train.py configs/hrda/goji26IRRG2goji30IRRG_hrda_3vi_phy_struct_gate.py

echo "===== Start task 2/6: goji30IRRG -> goji26IRRG (hrda, 3vi_phy_struct_gate) ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate.py

echo "===== Start task 3/6: goji26IRRG -> goji30RGB (hrda, 3vi_phy_struct_gate) ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate.py

echo "===== Start task 4/6: goji30RGB -> goji26IRRG (hrda, 3vi_phy_struct_gate) ====="
python tools/train.py configs/hrda/goji30RGB2goji26IRRG_hrda_3vi_phy_struct_gate.py

echo "===== Start task 5/6: goji26RGB -> goji30RGB (hrda, 3vi_phy_struct_gate) ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py

echo "===== Start task 6/6: goji30RGB -> goji26RGB (hrda, 3vi_phy_struct_gate) ====="
python tools/train.py configs/hrda/goji30RGB2goji26RGB_hrda_3vi_phy_struct_gate.py

echo "All hrda 3vi_phy_struct_gate training tasks finished."
