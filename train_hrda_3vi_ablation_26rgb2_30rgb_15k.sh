#!/bin/bash
set -e

echo "===== 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi.py

echo "===== 2/8 Physiology ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_phy_gate.py

echo "===== 3/8 Structural ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_struct_gate.py

echo "===== 4/8 MPARC ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py

echo "===== 5/8 Prototype ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_ps_proto.py

echo "===== 6/8 Edge ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_struct_edge.py

echo "===== 7/8 PBCR ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_pbcr.py

echo "===== 8/8 MPARC + PBCR ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py

echo "All 26RGB -> 30RGB HRDA+3VI aligned 20k ablations finished."

echo "===== 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi.py

echo "===== 2/8 Physiology ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_phy_gate.py

echo "===== 3/8 Structural ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_struct_gate.py

echo "===== 4/8 MPARC ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate.py

echo "===== 5/8 Prototype ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_ps_proto.py

echo "===== 6/8 Edge ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_struct_edge.py

echo "===== 7/8 PBCR ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_pbcr.py

echo "===== 8/8 MPARC + PBCR ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py

echo "All 26IRRG -> 30RGB HRDA+3VI aligned 20k ablations finished."

echo "===== 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi.py

echo "===== 2/8 Physiology ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_phy_gate.py

echo "===== 3/8 Structural ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_struct_gate.py

echo "===== 4/8 MPARC ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate.py

echo "===== 5/8 Prototype ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_ps_proto.py

echo "===== 6/8 Edge ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_struct_edge.py

echo "===== 7/8 PBCR ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_pbcr.py

echo "===== 8/8 MPARC + PBCR ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py

echo "All 30IRRG -> 26IRRG HRDA+3VI aligned 20k ablations finished."

echo "===== 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/pssal/goji26IRRG_hrda_3vi.py

echo "===== 2/8 Physiology ====="
python tools/train.py configs/hrda/pssal/goji26IRRG_hrda_3vi_phy_gate.py

echo "===== 3/8 Structural ====="
python tools/train.py configs/hrda/pssal/goji26IRRG_hrda_3vi_struct_gate.py

echo "===== 4/8 MPARC ====="
python tools/train.py configs/hrda/pssal/goji26IRRG_hrda_3vi_phy_struct_gate.py

echo "===== 5/8 Prototype ====="
python tools/train.py configs/hrda/pssal/goji26IRRG_hrda_3vi_ps_proto.py

echo "===== 6/8 Edge ====="
python tools/train.py configs/hrda/pssal/goji26IRRG_hrda_3vi_struct_edge.py

echo "===== 7/8 PBCR ====="
python tools/train.py configs/hrda/pssal/goji26IRRG_hrda_3vi_pbcr.py

echo "===== 8/8 MPARC + PBCR ====="
python tools/train.py configs/hrda/pssal/goji26IRRG_hrda_3vi_proto_struct_edge.py

echo "All goji26IRRG supervised HRDA+3VI aligned 20k ablations finished."

echo "===== 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/pssal/goji30RGB_hrda_3vi.py

echo "===== 2/8 Physiology ====="
python tools/train.py configs/hrda/pssal/goji30RGB_hrda_3vi_phy_gate.py

echo "===== 3/8 Structural ====="
python tools/train.py configs/hrda/pssal/goji30RGB_hrda_3vi_struct_gate.py

echo "===== 4/8 MPARC ====="
python tools/train.py configs/hrda/pssal/goji30RGB_hrda_3vi_phy_struct_gate.py

echo "===== 5/8 Prototype ====="
python tools/train.py configs/hrda/pssal/goji30RGB_hrda_3vi_ps_proto.py

echo "===== 6/8 Edge ====="
python tools/train.py configs/hrda/pssal/goji30RGB_hrda_3vi_struct_edge.py

echo "===== 7/8 PBCR ====="
python tools/train.py configs/hrda/pssal/goji30RGB_hrda_3vi_pbcr.py

echo "===== 8/8 MPARC + PBCR ====="
python tools/train.py configs/hrda/pssal/goji30RGB_hrda_3vi_proto_struct_edge.py

echo "All goji30RGB supervised HRDA+3VI aligned 20k ablations finished."
