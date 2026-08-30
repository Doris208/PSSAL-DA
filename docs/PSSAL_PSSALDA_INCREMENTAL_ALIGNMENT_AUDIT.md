# PSSAL / PSSAL-DA Incremental Alignment Audit

Scope: local code/config preparation only. Remote logs, checkpoints, best
iterations, and metrics are not available in this checkout and are not inferred.

Git snapshot at audit start:

- Branch: `doris`
- HEAD: `9ec6fc9a`
- Worktree note: many pre-existing inference-result adds/renames under
  `HRDA_struct_edge_infer` / `HRDA_SG_BDE_infer` were present and were not
  touched by this alignment work.

## Method Authority

The current Word method defines:

- `MPARC`: Physiology-Aware Prior Representation plus Spatial-Structural Prior
  Interaction.
- `F_t_PS/P_t_PS`: spatial-structural response interacts with decoder features
  before the segmentation classifier.
- `PBCR`: Prototype-Guided Semantic Refinement plus Structure-Guided Boundary
  Detail Enhancement.
- Word-version SG-BDE uses only `E_prob`, `E_img`, and `R_pair`; old layout and
  neighbor-patch responses are not part of the current paper path.

## Implementation Changes Made

- Added `edge_weight_mode='paper_pbcr'` in `mmseg/models/uda/dacs.py`.
- Added `struct_prior_response` forwarding through:
  `EncoderDecoder`, `HRDAEncoderDecoder`, `BaseDecodeHead`, `HRDAHead`, and
  `DAFormerHead`.
- Added `DAFormerHead.forward_features()` plus feature residual interaction:
  `x_ps = x + gamma * Conv1x1(R_struct * x)`, followed by the normal classifier.
- Enabled this feature path in `configs/hrda/pssalda_paper_pbcr_common.py` for
  the `_new` PSSAL-DA configs.
- Added `configs/hrda/pssalda_paper_pbcr_common.py` as a shared config helper.
- Added six-direction `struct_gate_new` configs.
- Added six-direction `phy_struct_gate_new` configs.
- Added six-direction `ps_proto_new` configs.
- Added six-direction `mparc_boundary_new` configs.
- Added six-direction `proto_struct_edge_new` configs. These are the
  Word-aligned continuation of the user's final `proto_struct_edge` experiment,
  not the old `proto_full_edge` ablation.
- Added `mmseg/models/uda/pssal_supervised.py` as a real supervised PSSAL
  training wrapper.
- Added supervised PSSAL configs under `configs/hrda/pssal/`:
  7 experiment settings x 7 ablations = 49 runnable overlay configs.
- Added seven supervised batch scripts:
  `train_pssal_00_supervised_3vi_control.sh` through
  `train_pssal_06_supervised_full_pssal.sh`.

## Existing Experiment Audit

| Original experiment | Original config path | Main implementation | Paper role | Current status | Notes |
|---|---|---|---|---|---|
| HRDA baseline | `configs/hrda/*_hrda.py` | DACS | UDA baseline | preserved | Baseline role only, not MPARC. |
| HRDA + 3VI | `configs/hrda/*_hrda_3vi.py` | DACS + 3VI input | 3VI control | preserved | Control for module ablation. |
| Physiology gate | `configs/hrda/*_hrda_3vi_phy_gate.py` | `PhyReliabilityHead`, DACS | physiology prior | preserved | Existing pseudo-label reliability path. |
| Structure gate | `configs/hrda/*_hrda_3vi_struct_gate.py` | `StructPriorConsistencyGate`, DACS | spatial-structural prior | old config preserved; `*_struct_gate_new.py` has feature interaction | Old files are not overwritten. |
| Physiology + structure | `configs/hrda/*_hrda_3vi_phy_struct_gate.py` | classwise joint gate | MPARC | old config preserved; `*_phy_struct_gate_new.py` has feature interaction | Existing joint gate remains unchanged. |
| Proto gate | `configs/hrda/*_hrda_3vi_ps_proto.py` | `ProtoGate`, DACS | MPARC + prototype | old config preserved; `*_ps_proto_new.py` has feature interaction | Existing ProtoGate retained for continuity. |
| Old proto_struct_edge | `configs/hrda/*_hrda_3vi_proto_struct_edge.py` | DACS + old SG-BDE helpers | final old experiment | preserved | Old final experiment remains untouched. |
| MPARC + boundary new | `configs/hrda/*_hrda_3vi_mparc_boundary_new.py` | MPARC-like path + Word PBCR edge | MPARC + SG-BDE | added | Prototype disabled; Word SG-BDE enabled. |
| Proto-struct-edge new | `configs/hrda/*_hrda_3vi_proto_struct_edge_new.py` | MPARC-like path + ProtoGate + Word PBCR edge | final PSSAL-DA candidate | added | Uses `proto_struct_edge_new` naming, not `full_pssalda_new`. |

## Supervised PSSAL Status

The supervised path is now implemented through `PSSALSupervised`, not by
disabling UDA with `use_uda=False`. This matters because the original
physiology, structure, prototype, and boundary modules live outside a plain
supervised HRDA segmentor.

`PSSALSupervised` uses:

- GT labels for normal segmentation loss.
- Optional physiology loss from `vi_img`.
- GT-labelled structural prior response for `F_t_PS/P_t_PS`.
- Source-only prototype update/loss for the supervised prototype ablations.
- GT-derived PBCR boundary weights for the supervised boundary ablations.

## Data and Label Mapping

- `GojiDataset` uses `reduce_zero_label=True`, so raw label `0` is ignored and
  raw labels `1-5` map to train IDs `0-4`.
- The existing class routing is consistent with the paper assumption:
  physiology classes `[0, 1]`; structural classes `[2, 3, 4]`.
- 3VI order in configs is `ndvi`, `ndre`, `cire`.

## Validation Notes

Static checks performed by this task are recorded in
`docs/RETRAINING_MATRIX.md`. Full training and model construction were not run
locally because this Windows environment lacks `torch`, `mmcv`, and `bash`.
