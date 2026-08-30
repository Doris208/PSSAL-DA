#!/bin/bash
set -e

echo "PSSAL-DA 1/3: ZN 26 IRRG to HN 30 IRRG"
python tools/train.py configs/hrda/goji26IRRG2goji30IRRG_hrda_3vi_proto_struct_edge_new.py

echo "PSSAL-DA 2/3: ZN 26 IRRG to HN 30 RGB"
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge_new.py

echo "PSSAL-DA 3/3: ZN 26 RGB to HN 30 RGB"
python tools/train.py configs/hrda/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge_new.py
