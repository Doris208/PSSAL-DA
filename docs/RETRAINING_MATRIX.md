# Retraining Matrix

Cell meanings:

- `REUSE`: code/config can be reused; this does not assert a local result
  exists.
- `RETRAIN`: new or `_new` configuration must be trained before reporting.
- `NEW`: newly added experiment node.
- `N/A`: not applicable.

## PSSAL-DA Six-Direction Matrix

| Experiment step | 26IRRG->30IRRG | 30IRRG->26IRRG | 26IRRG->30RGB | 30RGB->26IRRG | 26RGB->30RGB | 30RGB->26RGB | Reason | Script |
|---|---|---|---|---|---|---|---|---|
| 00_hrda_3vi_control | REUSE | REUSE | REUSE | REUSE | REUSE | REUSE | Existing 3VI DACS configs. | existing |
| 01_physiology_prior | REUSE | REUSE | REUSE | REUSE | REUSE | REUSE | Existing physiology gate configs preserved. | `train_hrda_3vi_phy_gate.sh` |
| 02_spatial_structural_prior | RETRAIN | RETRAIN | RETRAIN | RETRAIN | RETRAIN | RETRAIN | `F_t_PS/P_t_PS` feature interaction is newly implemented. | `train_hrda_3vi_struct_gate_new.sh` |
| 03_mparc | RETRAIN | RETRAIN | RETRAIN | RETRAIN | RETRAIN | RETRAIN | MPARC paper path now includes feature interaction. | `train_hrda_3vi_phy_struct_gate_new.sh` |
| 04_mparc_plus_prototype | RETRAIN | RETRAIN | RETRAIN | RETRAIN | RETRAIN | RETRAIN | Existing ProtoGate retained, feature interaction added in aligned path. | `train_hrda_3vi_ps_proto_new.sh` |
| 05_mparc_plus_boundary | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | Missing node; prototype disabled and Word SG-BDE enabled. | `train_hrda_3vi_mparc_boundary_new.sh` |
| 06_proto_struct_edge_final | RETRAIN | RETRAIN | RETRAIN | RETRAIN | RETRAIN | RETRAIN | Existing final `proto_struct_edge` path is preserved; `_new` uses current Word feature interaction and SG-BDE. | `train_hrda_3vi_proto_struct_edge_new.sh` |

## New PSSAL-DA Configs

| Direction | Struct config | MPARC config | MPARC + proto config | MPARC + boundary config | Full PSSAL-DA config |
|---|---|---|---|---|---|
| 26IRRG->30IRRG | `configs/hrda/goji26IRRG2goji30IRRG_hrda_3vi_struct_gate_new.py` | `configs/hrda/goji26IRRG2goji30IRRG_hrda_3vi_phy_struct_gate_new.py` | `configs/hrda/goji26IRRG2goji30IRRG_hrda_3vi_ps_proto_new.py` | `configs/hrda/goji26IRRG2goji30IRRG_hrda_3vi_mparc_boundary_new.py` | `configs/hrda/goji26IRRG2goji30IRRG_hrda_3vi_proto_struct_edge_new.py` |
| 30IRRG->26IRRG | `configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_struct_gate_new.py` | `configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate_new.py` | `configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_ps_proto_new.py` | `configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_mparc_boundary_new.py` | `configs/hrda/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge_new.py` |
| 26IRRG->30RGB | `configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_struct_gate_new.py` | `configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate_new.py` | `configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_ps_proto_new.py` | `configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_mparc_boundary_new.py` | `configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge_new.py` |
| 30RGB->26IRRG | `configs/hrda/goji30RGB2goji26IRRG_hrda_3vi_struct_gate_new.py` | `configs/hrda/goji30RGB2goji26IRRG_hrda_3vi_phy_struct_gate_new.py` | `configs/hrda/goji30RGB2goji26IRRG_hrda_3vi_ps_proto_new.py` | `configs/hrda/goji30RGB2goji26IRRG_hrda_3vi_mparc_boundary_new.py` | `configs/hrda/goji30RGB2goji26IRRG_hrda_3vi_proto_struct_edge_new.py` |
| 26RGB->30RGB | `configs/hrda/goji26RGB2goji30RGB_hrda_3vi_struct_gate_new.py` | `configs/hrda/goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate_new.py` | `configs/hrda/goji26RGB2goji30RGB_hrda_3vi_ps_proto_new.py` | `configs/hrda/goji26RGB2goji30RGB_hrda_3vi_mparc_boundary_new.py` | `configs/hrda/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge_new.py` |
| 30RGB->26RGB | `configs/hrda/goji30RGB2goji26RGB_hrda_3vi_struct_gate_new.py` | `configs/hrda/goji30RGB2goji26RGB_hrda_3vi_phy_struct_gate_new.py` | `configs/hrda/goji30RGB2goji26RGB_hrda_3vi_ps_proto_new.py` | `configs/hrda/goji30RGB2goji26RGB_hrda_3vi_mparc_boundary_new.py` | `configs/hrda/goji30RGB2goji26RGB_hrda_3vi_proto_struct_edge_new.py` |

## PSSAL Supervised Matrix

| Experiment step | 26IRRG | 30RGB | 26RGB | 30IRRG | 30IRRG->26IRRG | 26IRRG->30RGB | 26RGB->30RGB | Script |
|---|---|---|---|---|---|---|---|---|
| 00_supervised_3vi_control | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | `train_pssal_00_supervised_3vi_control.sh` |
| 01_supervised_physiology_prior | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | `train_pssal_01_supervised_physiology_prior.sh` |
| 02_supervised_spatial_structural_prior | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | `train_pssal_02_supervised_spatial_structural_prior.sh` |
| 03_supervised_mparc | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | `train_pssal_03_supervised_mparc.sh` |
| 04_supervised_mparc_plus_prototype | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | `train_pssal_04_supervised_mparc_plus_prototype.sh` |
| 05_supervised_mparc_plus_boundary | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | `train_pssal_05_supervised_mparc_plus_boundary.sh` |
| 06_supervised_full_pssal | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | NEW/RETRAIN | `train_pssal_06_supervised_full_pssal.sh` |

## Validation Record

- Python syntax: passed for all modified model/UDA files through
  `python -m py_compile`.
- PSSAL supervised config parse: passed AST parsing for all 63 files under
  `configs/hrda/pssal/`.
- PSSAL supervised script path check: passed; all 7 supervised scripts point to
  existing configs and each script has 7 training commands.
- PSSAL-DA config parse: passed AST parsing for all 30 `_new` configs.
- PSSAL-DA script path check: passed; all 5 `_new` scripts point to existing
  configs and each script has 6 training commands.
- Full MMCV config parse: not run because this local environment reports
  `ModuleNotFoundError: No module named 'mmcv'`.
- Dummy model forward / loss forward: not run because this local environment
  reports `ModuleNotFoundError: No module named 'torch'`.
- Bash syntax: not run because this PowerShell environment cannot find `bash`.
