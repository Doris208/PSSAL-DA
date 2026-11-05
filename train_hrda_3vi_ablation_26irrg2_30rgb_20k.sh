#!/bin/bash
set -e



echo "===== 8/8 MPARC + PBCR ====="
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py

echo "All 26IRRG -> 30RGB HRDA+3VI aligned 20k ablations finished."


echo "===== 8/8 MPARC + PBCR ====="
python tools/train.py configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py

echo "All 30IRRG -> 26IRRG HRDA+3VI aligned 20k ablations finished."


echo "===== 8/8 MPARC + PBCR ====="
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py