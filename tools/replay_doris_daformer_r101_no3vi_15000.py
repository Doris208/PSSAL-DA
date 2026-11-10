"""Replay the DAFormer R101 RGB-only Baseline Best mIoU checkpoint at Iter 15000."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools.replay_doris_baseline_18000 import main as replay_baseline

CHECKPOINT = Path('/root/shared-nvme/work_dirs/doris_no3vi_r101/goji26RGB2goji30RGB_daformer_r101_no3vi/best_mIoU_iter_15000.pth')


def main(argv=None):
    return replay_baseline(argv,default_checkpoint=CHECKPOINT)


if __name__=='__main__':
    sys.exit(main())
