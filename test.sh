# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add HR, LR, and attention visualization
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: bash test.sh RUN_DIR [tools.test options, e.g. --show-dir DIR]" >&2
    exit 2
fi
TEST_ROOT=$1
shift
shopt -s nullglob
CONFIG_FILES=("${TEST_ROOT}"/*"${TEST_ROOT: -1}".json)
if [ "${#CONFIG_FILES[@]}" -eq 0 ]; then
    for candidate in "${TEST_ROOT}"/*.py; do
        if [[ "${candidate##*/}" != source_* ]]; then
            CONFIG_FILES+=("$candidate")
        fi
    done
fi
if [ "${#CONFIG_FILES[@]}" -ne 1 ]; then
    echo "Expected one effective config in ${TEST_ROOT}; use tools/test.py for an explicit config." >&2
    exit 2
fi
CONFIG_FILE=${CONFIG_FILES[0]}
CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
echo "Config File: ${CONFIG_FILE}"
echo "Checkpoint File: ${CHECKPOINT_FILE}"
# Evaluation only by default. Export masks explicitly with --show-dir DIR.
python -m tools.test "${CONFIG_FILE}" "${CHECKPOINT_FILE}" --eval mIoU "$@"

# Uncomment the following lines to visualize the LR predictions,
# HR predictions, or scale attentions of HRDA:
#python -m tools.test "${CONFIG_FILE}" "${CHECKPOINT_FILE}" --show-dir "${TEST_ROOT}/preds_LR" --opacity 1 --hrda-out LR
#python -m tools.test "${CONFIG_FILE}" "${CHECKPOINT_FILE}" --show-dir "${TEST_ROOT}/preds_HR" --opacity 1 --hrda-out HR
#python -m tools.test "${CONFIG_FILE}" "${CHECKPOINT_FILE}" --show-dir "${TEST_ROOT}/preds_ATT" --opacity 1 --hrda-out ATT
