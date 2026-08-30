# Copyright (c) OpenMMLab. All rights reserved.

GOJI_CLASSES = ('goji', 'vegetation', 'building', 'water', 'road')

GOJI_PALETTE = [[220, 20, 60], [34, 139, 34], [70, 70, 70], [0, 0, 255],
                [244, 164, 96]]

# Raw labels stored in ann_dir patches follow the Vaihingen-style pipeline:
# 0 is ignore/background and 1..5 are valid class ids.
GOJI_RAW_ID_TO_COLOR = [[0, 0, 0]] + GOJI_PALETTE

# Reduced labels are produced after reduce_zero_label:
# 255 is ignore and 0..4 map to the valid classes.
GOJI_REDUCED_ID_TO_COLOR = GOJI_PALETTE
