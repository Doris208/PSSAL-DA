# PSSAL-DA

Official initial implementation of **PSSAL-DA: Physiology-Structural Semantic Awareness Learning for Domain-Adaptive Semantic Segmentation of Goji Cultivation Areas Using Sentinel-2 Multispectral Data**.

PSSAL-DA addresses cross-regional and cross-modal semantic segmentation of goji cultivation areas. It extends a teacher-student UDA framework with two task-specific components:

- **Multi-Prior-Aware Representation Construction (MPARC)** integrates NDVI, NDRE, and CIRE physiological responses with spatial-structural priors for feature interaction and target pseudo-supervision.
- **Prototype-Boundary Collaborative Refinement (PBCR)** combines cross-domain class prototypes with structure-guided boundary supervision to improve semantic consistency and local class transitions.

This repository is an initial research release accompanying a manuscript currently under review. It contains the main PSSAL-DA implementation and the three transfer configurations reported in the manuscript. Training checkpoints, experiment logs, prediction images, comparison-method implementations, and unpublished result artifacts are intentionally excluded.

## Repository Structure

```text
PSSAL-DA/
├── configs/                  Main PSSAL-DA configurations and required bases
├── mmseg/                    Segmentation framework and PSSAL-DA modules
├── resources/                Third-party license texts
├── tools/                    Training and evaluation entry points
├── train_pssal_da.sh         Three manuscript transfer tasks
├── LICENSE
└── LICENSES.md
```

The retained `hrda` names inside configuration paths and several framework classes are compatibility identifiers inherited from the upstream implementation. They are kept to avoid unnecessary breakage and to preserve proper attribution; the public training entry point and project documentation use PSSAL-DA terminology.

## Environment

The code follows the environment used by the upstream HRDA implementation:

```bash
conda create -n pssal-da python=3.8.5 -y
conda activate pssal-da
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7
```

Download the MiT-B5 ImageNet pretrained weights provided by [SegFormer](https://github.com/NVlabs/SegFormer) and place the file at:

```text
pretrained/mit_b5.pth
```

Run all commands from the repository root. CUDA, PyTorch, and MMCV builds must be mutually compatible with the target system.

## Dataset

NGMSeg is not included in this initial code release. Place an authorized local copy at:

```text
shared-nvme/mydataset/
├── 26/
└── 30/
```

The five semantic classes are goji, vegetation, building, water, and road. Existing subdirectory names and file names are part of the data-loading contract and should not be changed. The `shared-nvme/` directory is ignored by Git so that local data cannot be committed accidentally.

The Apache-2.0 code license does not automatically grant rights to the dataset. Dataset-use terms will be provided separately by the dataset authors.

## Training

Train the three transfer settings reported in the manuscript:

```bash
bash train_pssal_da.sh
```

The script runs:

1. ZN 26 (IRRG) to HN 30 (IRRG)
2. ZN 26 (IRRG) to HN 30 (RGB)
3. ZN 26 (RGB) to HN 30 (RGB)

To launch one setting independently:

```bash
python tools/train.py configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge_new.py
```

Logs and checkpoints are written under `work_dirs/`, which is excluded from version control.

## Evaluation

```bash
python tools/test.py CONFIG CHECKPOINT --eval mIoU
```

Replace `CONFIG` with one of the three main configuration files and `CHECKPOINT` with a locally trained checkpoint.

## Main Implementation Files

- `mmseg/models/uda/phy_reliability.py`: physiology-aware response modeling.
- `mmseg/models/uda/struct_prior_gate.py`: spatial-structural prior modeling and interaction.
- `mmseg/models/uda/proto_gate.py`: cross-domain prototype construction and refinement.
- `mmseg/models/losses/struct_edge_loss.py`: structure-guided boundary supervision.
- `mmseg/models/uda/dacs.py`: integration of PSSAL-DA into teacher-student self-training.
- `configs/hrda/pssalda_paper_pbcr_common.py`: the PBCR configuration used by the main experiments.

## Acknowledgements

PSSAL-DA is built on [HRDA](https://github.com/lhoyer/HRDA), [DAFormer](https://github.com/lhoyer/DAFormer), [DACS](https://github.com/vikolss/DACS), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). The relevant copyright and license notices are retained in the source files and summarized in [LICENSES.md](LICENSES.md).

## Citation

Citation information will be added after the manuscript is accepted.

## License

The code is released under the [Apache License 2.0](LICENSE), subject to the third-party notices in [LICENSES.md](LICENSES.md). The dataset is not covered by the code license.
