# Paper Experiment Mapping

## Input Prior Study

| Experiment layer | Original experiment | Original config | Paper role | Reuse or new | Script | Notes |
|---|---|---|---|---|---|---|
| input prior study | HRDA baseline | `configs/hrda/*_hrda.py` | UDA baseline | reuse | existing scripts | No VI input. |
| input prior study | HRDA + NDVI | `configs/hrda/*_hrda_ndvi.py` | single-index input study | reuse | existing scripts | Input prior only. |
| input prior study | HRDA + 3VI | `configs/hrda/*_hrda_3vi.py` | 3VI control | reuse | existing scripts | Base for module ablation. |

## PSSAL-DA Module Ablation

| Experiment layer | Original experiment | Paper role | Reuse or new | New config | Script | Notes |
|---|---|---|---|---|---|---|
| 00 | HRDA + 3VI | 3VI control | reuse | N/A | existing | DACS baseline with 3VI. |
| 01 | physiology gate | physiology prior | reuse | N/A | `train_hrda_3vi_phy_gate.sh` | Existing path preserved. |
| 02 | structure gate | spatial-structural prior | new aligned path for paper rerun | `configs/hrda/*_hrda_3vi_struct_gate_new.py` | `train_hrda_3vi_struct_gate_new.sh` | `F_t_PS/P_t_PS` is now available through `struct_prior_interaction`. |
| 03 | phy+struct joint | MPARC | new aligned path for paper rerun | `configs/hrda/*_hrda_3vi_phy_struct_gate_new.py` | `train_hrda_3vi_phy_struct_gate_new.sh` | Old joint gate configs preserved. |
| 04 | ps_proto | MPARC + prototype | new aligned path for paper rerun | `configs/hrda/*_hrda_3vi_ps_proto_new.py` | `train_hrda_3vi_ps_proto_new.sh` | Existing ProtoGate retained. |
| 05 | missing old exact node | MPARC + boundary | new | `configs/hrda/*_hrda_3vi_mparc_boundary_new.py` | `train_hrda_3vi_mparc_boundary_new.sh` | Prototype disabled; Word SG-BDE enabled. |
| 06 | proto_struct_edge | final PSSAL-DA candidate | new Word-aligned version | `configs/hrda/*_hrda_3vi_proto_struct_edge_new.py` | `train_hrda_3vi_proto_struct_edge_new.sh` | This is the aligned version of final `proto_struct_edge`, not old `proto_full_edge`. |

## PSSAL Supervised Ablation

Seven supervised settings are generated:

- Same-area: `goji26IRRG`, `goji30RGB`, `goji26RGB`, `goji30IRRG`.
- Cross-domain supervised train/eval settings:
  `goji30IRRG2goji26IRRG`, `goji26IRRG2goji30RGB`,
  `goji26RGB2goji30RGB`.

| Experiment layer | Paper role | Config pattern | Script |
|---|---|---|---|
| 00 | supervised 3VI control | `configs/hrda/pssal/*_00_supervised_3vi_control.py` | `train_pssal_00_supervised_3vi_control.sh` |
| 01 | supervised physiology prior | `configs/hrda/pssal/*_01_supervised_physiology_prior.py` | `train_pssal_01_supervised_physiology_prior.sh` |
| 02 | supervised spatial-structural prior | `configs/hrda/pssal/*_02_supervised_spatial_structural_prior.py` | `train_pssal_02_supervised_spatial_structural_prior.sh` |
| 03 | supervised MPARC | `configs/hrda/pssal/*_03_supervised_mparc.py` | `train_pssal_03_supervised_mparc.sh` |
| 04 | supervised MPARC + prototype | `configs/hrda/pssal/*_04_supervised_mparc_plus_prototype.py` | `train_pssal_04_supervised_mparc_plus_prototype.sh` |
| 05 | supervised MPARC + boundary | `configs/hrda/pssal/*_05_supervised_mparc_plus_boundary.py` | `train_pssal_05_supervised_mparc_plus_boundary.sh` |
| 06 | supervised full PSSAL | `configs/hrda/pssal/*_06_supervised_full_pssal.py` | `train_pssal_06_supervised_full_pssal.sh` |

The supervised section is implemented through `PSSALSupervised`, so these
configs execute the PSSAL modules instead of only running plain supervised HRDA.
