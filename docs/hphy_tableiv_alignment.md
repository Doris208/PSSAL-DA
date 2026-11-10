# H-Phy Table IV alignment and local acceptance

## Experiment identity

Main direction is ZN_26 RGB → HN_30 RGB, `goji26RGB2goji30RGB`. All eight entries use MiT-B5, six input channels, seed 0, the same datasets/splits/normalization, AdamW at 6e-5, inherited Poly warmup, validation every 1000 iterations and a 15000-iteration runner. The H-Phy VI input remains separate and three-channel. Config inheritance is parsed by real MMCV and checked by `tools/check_doris_hphy_configs.py`.

Historical 20K results remain historical reports. The new table requires eight fresh 15K runs before it can report comparable accuracy. Formal 15K training has not been started. Local-only development and manual upload by the author apply to this delivery.

## Eight-row configuration audit

| Row | Experiment | Phy | Structural | Prototype | Edge | Config |
|---|---|---|---|---|---|---|
| 1/8 | Baseline | Off | Off | Off | Off | `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py` |
| 2/8 | Physiology | H-Phy | Off | Off | Off | `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py` |
| 3/8 | Structural | Off | On | Off | Off | `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_gate.py` |
| 4/8 | MPARC | H-Phy | On | Off | Off | `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py` |
| 5/8 | Prototype | Off | Off | On | Off | `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_ps_proto.py` |
| 6/8 | Edge | Off | Off | Off | On | `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_edge.py` |
| 7/8 | PBCR | Off | Off | On | On | `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_pbcr.py` |
| 8/8 | Full | H-Phy | On | On | On | `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py` |

The Baseline, Structural, Prototype, Edge and PBCR config files retain their original contents. Their historical default `work_dir` fields are preserved. The launcher overrides those defaults with fresh destinations. Every H-Phy leaf also has an isolated new default directory for direct `tools/train.py` use.

### Actual launcher destinations

- 1/8 Baseline uses `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi`.
- 2/8 Physiology uses `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_gate`.
- 3/8 Structural uses `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_gate`.
- 4/8 MPARC uses `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate`.
- 5/8 Prototype uses `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_ps_proto`.
- 6/8 Edge uses `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_edge`.
- 7/8 PBCR uses `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_pbcr`.
- 8/8 Full uses `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge`.

The script follows the original flat eight-section layout. Each echo is followed by a complete `python tools/train.py` command. Running the script without arguments runs all eight groups. Copy one Python command to run one group. Configuration checks use `python tools/check_doris_hphy_configs.py --all-directions`.

Each work-directory basename matches the original stage1 script. Only the output root changes to `doris_hphy_early4k`. The training entry point creates independent logs inside each experiment directory and rejects occupied runs. The shell uses `set -e` to stop on a failed Python command.

## Nine H-Phy leaf defaults

All entries below use `hierarchical_gv`, beta 1, source/target coefficients 0.08/0.008, direct target GV coefficient 0, confidence threshold 0.968, gate floor 0.75, no filter threshold, detached gate and early_nonzero_linear scheduling with initial strength 0.10 and full strength at 4000. Their runner endpoint is 15000. Existing structure/prototype/edge coefficients remain direction-specific.

| Configuration | Resolved default work directory |
|---|---|
| `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py` | `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_gate_hphy_early4k` |
| `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py` | `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate_hphy_early4k` |
| `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py` | `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge_hphy_early4k` |
| `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_gate.py` | `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_phy_gate_hphy_early4k` |
| `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate.py` | `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate_hphy_early4k` |
| `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py` | `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge_hphy_early4k` |
| `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_gate.py` | `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_phy_gate_hphy_early4k` |
| `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate.py` | `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate_hphy_early4k` |
| `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py` | `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge_hphy_early4k` |

## Execution commands after manual upload

Run these from the uploaded project root in the existing Linux training environment.

```bash
# Parse/check all eight jobs. This starts no training.
python tools/check_doris_hphy_configs.py --all-directions

# Check or execute one group. These examples use Physiology, row 2.
python tools/check_doris_hphy_configs.py --experiment 2
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_gate"

# Execute eight fresh jobs sequentially.
bash scripts/run_doris_tableiv_26rgb_to_30rgb_hphy15k.sh

# Complete real GPU acceptance before formal training. Use a new output each time.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/smoke_doris_hphy.py --device cuda:0 --experiments 2 4 8   --output /root/shared-nvme/work_dirs/doris_hphy_early4k/smoke_01
python tools/check_doris_hphy_configs.py --all-directions
python -m pytest tests -q -rs
```

A single-row run consumes that row's reserved destination. Running `all` afterward correctly refuses the occupied destination. Use the remaining row numbers to continue other fresh groups. Explicitly resuming a completed/partial new run uses the training entry point directly.

```bash
cfg=configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py
wd=/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_gate
python tools/train.py "$cfg" --work-dir "$wd" --resume-from "$wd/latest.pth"

# Read-only weight replay, after the corresponding checkpoints exist.
python tools/replay_doris_mechanisms.py "$cfg" --work-dir "$wd"   --iterations 2000 4000 6000 10000 15000 --device cuda:0
```

An optional fresh legacy 15K control uses the existing entry with explicit legacy overrides. It is separate from the eight-row primary table and from historical 20K results.

```bash
legacy=/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/legacy15k_control
if [ -e "$legacy" ]; then echo "Existing legacy control directory" >&2; exit 1; fi
python tools/train.py "$cfg" --work-dir "$legacy" --seed 0 --options   uda.phy_cfg.mode=legacy_binary uda.phy_cfg.phy_min_weight=0.82   uda.phy_cfg.phy_filter_threshold=0.30   uda.phy_cfg.phy_warmup_iters=6000 uda.phy_cfg.phy_rampup_iters=4000   uda.phy_cfg.phy_gate_schedule=legacy_warmup_linear hphy_version=legacy_binary_control   name=doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate
```

### Offline evidence and sampling

Main spatial evidence uses the pre-existing fixed target `3_1024_1280_1280_1536.tif`. All checkpoints retain the same sample and 30-pixel valid margins. Multi-tile reliability analysis requires an author-fixed list including representative failures, declared before inspecting the results. Other directions retain their own existing samples and are supplementary to the main direction.

`--phy-evidence` explicitly adds offline reference labels after model inference. `--target-filenames` accepts a preselected validation TIFF list. Replay preserves the five-class mapping, weights, geometry and source hashes. Run one checkpoint over the declared list and aggregate its reference-enriched NPZ files using `tools/doris_phy_evidence.py --arrays FILE1 FILE2 --output NEW_JSON`. Mixing checkpoint iterations/configurations and duplicate sample/iteration identities is rejected. The default replay keeps the original single fixed sample. Final Goji IoU, Vegetation IoU, Veg.mIoU and five-class mIoU require independent Student predictions on the unchanged evaluation split. Those performance measurements remain pending.

## Compatibility adjustments

- Name-matching in both fixed-probe selection and checkpoint policy accepts `_hphy15k`, `_hphy_early3k` and `_hphy_early4k`. This retains fixed sample identity, complete checkpoint scheduling and occupied-directory protection for the new configurations.
- Training logs are written by tools/train.py inside each experiment directory.
- DACS total-loss logging excludes the new raw components and duplicate target aliases. The minimal logger retains `hphy_*` class ratios, agreement/gate means and supported-weight attenuation statistics. Actual backward losses retain their explicit coefficients.
- Legacy response fixtures now explicitly construct legacy mode. New H-Phy tests exercise their own mode and checkpoints. Existing tests with obsolete 20K endpoints, retired config locations or historical active-command counts were corrected against the current baseline revision and current source.

## Earlier delayed-schedule local verification

The results in this section were recorded before the early nonzero schedule change. Current 4K acceptance is recorded in `docs/hphy_early4k_acceptance.md`.

- Real MMCV inheritance audit passed for eight main-direction entries plus the six affected leaves in the other directions. Protocol equality, module masks, structural interaction, boundary loss and Prototype thresholds were checked.
- Static compilation and Bash syntax checks passed. The launcher's real `all --check` path passed. Synthetic Bash execution tests cover all eight commands, a single Full run, collisions, missing configurations and immediate failure at row 2. No formal training command was run against the production destinations.
- Full real MiT-B5/HRDA/DACS CPU smoke passed for Physiology, MPARC and Full at schedule positions 6000 and 10000. Each completed source/main/mixed/auxiliary loss computation and backward, nonzero high-confidence target vegetation BCE, optimizer update, full checkpoint/optimizer roundtrip, read-only probe export and Student validation output.
- Synthetic smoke uses random model initialization with pretrained paths cleared within the smoke only. Controlled EMA classifier weights generate high-confidence nonvegetation at 6000 and Goji at 10000. The real Teacher forward and EMA update still execute. Formal configurations retain their original initialization paths.
- H-Phy head parameter count is 5202. Total DACS parameter counts, including EMA, are 171381606 for Physiology and 171512678 for MPARC/Full. Branch buffers are included in checkpoint tests.
- The existing real-TIF test requires `DORIS_PROBE_DATA_ROOT`. Required local TIFFs are absent. GPU is unavailable locally and remote GPU access is excluded by the author's instruction. These two acceptance items remain unverified. No complete 15K performance run or historical full-data replay was performed.
- Local test libraries are isolated under ignored `outputs/hphy_deps`. CPU acceptance used Python 3.8.20, PyTorch 2.3.0 and MMCV 1.4.0.

Detailed evidence resides in `outputs/hphy_full_cpu_smoke_controlled/report.json`, `outputs/hphy_config_audit_final.json`, `outputs/hphy_launcher_real_check.log` and the final pytest log. Full repository regression completed with **166 PASS, 1 SKIP** in `outputs/hphy_final_verified.log`. The skipped real-TIF test is unchanged and requires the absent dataset. Final targeted logging/checkpoint/launcher/evidence tests completed with **45 PASS** in `outputs/hphy_final_targeted.log`, including the final absolute-weight-budget correction. No failed test was converted to a skip.

Additional final integration checks passed for all eight entries with `doris_checkpoint_plan.applies`, a final checkpoint at 15000 and the exact main-direction fixed-probe identity.

## Complete changed-file list

- `.gitattributes`
- `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_gate.py`
- `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate.py`
- `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py`
- `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_gate.py`
- `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate.py`
- `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py`
- `docs/hphy_design_and_evidence.md`
- `docs/hphy_tableiv_alignment.md`
- `mmseg/models/uda/dacs.py`
- `mmseg/models/uda/phy_reliability.py`
- `mmseg/utils/mechanism.py`
- `mmseg/utils/precision_logger.py`
- `scripts/run_doris_tableiv_26rgb_to_30rgb_hphy15k.sh`
- `tests/test_doris_checkpoint_policy.py`
- `tests/test_doris_checkpoint_replay.py`
- `tests/test_doris_comparisons.py`
- `tests/test_doris_physiology.py`
- `tests/test_doris_stage1.py`
- `tests/test_hphy.py`
- `tests/test_hphy_launcher.py`
- `tests/test_minimal_output.py`
- `tools/build_doris_mechanism_figures.py`
- `tools/check_doris_hphy_configs.py`
- `tools/doris_checkpoint_plan.py`
- `tools/doris_phy_evidence.py`
- `tools/replay_doris_mechanisms.py`
- `tools/smoke_doris_hphy.py`
- `tools/train.py`
