#!/usr/bin/env bash
set -e

run_task() {
  local config="$1"
  local work_dir="$2"
  if [ -d "$work_dir" ] && [ -n "$(find "$work_dir" -mindepth 1 -print -quit)" ]; then
    echo "Refusing to run: work_dir exists and is not empty: $work_dir" >&2
    exit 1
  fi
  python tools/train.py "$config" --work-dir "$work_dir"
}

run_task \
  configs/memoryadaptnet/goji30IRRG2goji26IRRG_memoryadaptnet_r101_3vi.py \
  /root/shared-nvme/work_dirs/memoryadaptnet_r101/goji30IRRG2goji26IRRG_memoryadaptnet_r101_3vi

run_task \
  configs/memoryadaptnet/goji26IRRG2goji30RGB_memoryadaptnet_r101_3vi.py \
  /root/shared-nvme/work_dirs/memoryadaptnet_r101/goji26IRRG2goji30RGB_memoryadaptnet_r101_3vi

run_task \
  configs/memoryadaptnet/goji26RGB2goji30RGB_memoryadaptnet_r101_3vi.py \
  /root/shared-nvme/work_dirs/memoryadaptnet_r101/goji26RGB2goji30RGB_memoryadaptnet_r101_3vi
