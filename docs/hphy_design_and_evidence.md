# H-Phy V1 design and evidence

## Scope and inputs

Implementation starts from `ef089d4dc036b4b1ac3069d5b4e6964734be3da5` on local branch `codex/hphy15k`. Runtime training metadata records the actual Git SHA, resolved configuration, head version and physiology parameters. Uncommitted source changes require retaining this diff alongside the SHA.

The segmentation network keeps its original six-channel RGB/IRRG plus NDVI, NDRE and CIRE input. The independent H-Phy head receives only aligned `vi_img`, shaped `[B,3,H,W]`. `LoadGojiIndices` retains the existing clipping/normalization and `vi_fields` geometric augmentation. Its normalized VI input differs from the unnormalized indices appended to the segmentation input. Dataset and augmentation implementations are unchanged.

Reduced labels remain 0 Goji, 1 Vegetation, 2 Building, 3 Water, 4 Road and 255 Ignore. `UDADataset` returns source GT and target imagery/VI/validity to DACS. Target annotations are excluded from the training loss inputs. Offline reference analysis reads validation labels after inference.

## Computation and gradients

Two Conv-BN-ReLU blocks share a VI encoder with 16 and 32 channels. Independent 1×1 heads produce vegetation and conditional Goji logits. Each output has shape `[B,1,H,W]`.

```text
R_veg = sigmoid(logit_veg)
Q_G = sigmoid(logit_gv)
P_G = R_veg * Q_G
P_V = R_veg * (1-Q_G)
P_N = 1-R_veg
P_G + P_V + P_N = 1

L_veg_src = masked mean BCEWithLogits(logit_veg, GT in {0,1})
L_gv_src = equal present-group means of BCEWithLogits(logit_gv, GT==0)
           over GT==0 and GT==1 only
L_phy_src_raw = (L_veg_src + beta_gv*L_gv_src)/(1+beta_gv)
L_phy_src_weighted = 0.08*L_phy_src_raw

L_phy_tgt = 0.008*balanced BCEWithLogits(logit_veg, Teacher in {0,1})
           on valid Teacher predictions with confidence >= 0.968
L_gv_tgt = 0

S_phy = P_G if Teacher==0, P_V if Teacher==1, 1 otherwise
rho = 0.10 + 0.90*clip(local_iter/4000, 0, 1)
G_phy = 1-rho*(1-0.75)*(1-detach(S_phy))
        on spatial-valid Teacher vegetation predictions, 1 elsewhere
W_after_phy = W_before_phy*G_phy
attenuation_phy = 1-G_phy
```

Both BCE paths run in fp32. Missing class groups renormalize over present groups. Fully empty groups return graph-connected zero losses. Source losses update the VI encoder and the appropriate heads. Target vegetation BCE updates the shared encoder and vegetation head. It has no direct gradient into `gv_head`. Changes to the shared encoder can still change subsequent conditional responses. Detached gate weights block the target segmentation loss from updating H-Phy.

DACS retains its batch-wide high-confidence fraction times spatial-valid mask as its initial pseudo weight. Existing EMA, feature-distance loss and ClassMix remain in place. Physiology and structure are applied before the Prototype update/weight/loss inputs. Edge keeps the existing pre-Prototype snapshot interface and its original reliability calculation. Structure, Prototype and Edge formulas and coefficients are unchanged. The five phy-disabled Table IV configurations keep their original module definitions and textually identical configuration content after line-ending normalization.

Source diagnostics distinguish `loss_phy_src_veg_raw`, `loss_phy_src_gv_raw`, `loss_phy_src_raw` and `loss_phy_src_weighted`. Target `loss_phy_tgt` remains lambda-weighted, with explicit vegetation and zero direct-GV aliases. Total-loss reporting excludes raw and duplicate diagnostic fields. Legacy `loss_phy_src` remains raw.

## Compatibility and evidence

`phy_cfg.mode` selects `legacy_binary` or `hierarchical_gv`. An omitted mode selects legacy. The old head parameter keys and old target-loss arithmetic remain intact. H-Phy checkpoint loading requires the complete new head and matching tensor shapes, including through MMCV's non-strict loader. A legacy/H-Phy mode mismatch raises a readable error. Replay also compares saved and requested modes before construction. Historical H-Phy replay restores scheduling keys from the saved configuration. Other model/input checks remain strict. Legacy binary replay retains its historical configuration.

Online and replay share `branch_probe_payload`. H-Phy NPZ exports VI maps, Teacher labels/confidence, `r_veg`, `q_goji_given_veg`, three decomposed probabilities, `s_phy_class_agreement`, `g_phy`, pre/post physiology weights, `attenuation_phy` and `valid_mask`. Metadata identifies mode, class mapping, work directory and Teacher identity. `r_phy` is retained as a compatibility alias of `r_veg`. New main panels use `S_phy` and `1-G_phy`. Legacy paper-plate rendering rejects H-Phy arrays to prevent a misleading binary interpretation.

`tools/doris_phy_evidence.py` can aggregate reference-enriched NPZ files from multiple independently evaluated tiles at one checkpoint/configuration. It reports the four GT→Teacher Goji/Vegetation groups, means/medians/quantiles, error-detection ROC AUC and average precision, Teacher uncertainty comparisons, matched absolute pseudo-weight budgets with fractional tie groups and high-confidence errors. Evaluation uses valid reference pixels with positive pre-gate weights. Missing groups yield null metrics. Confidence intervals resample tiles and remain unavailable for a single tile. Spatial probes preserve the full input grid. The configured 256×256 tiles use the existing valid crop `[30:226,30:226]`.

## Manuscript changes and limits

- Replace the physiology schematic with a shared VI encoder and two labeled outputs, `R_veg` and `Q_G|veg`.
- Revise the text around equations (5), (7) and (8) to state the hierarchical decomposition, Teacher-argmax conditional agreement and detached continuous multiplicative gate above.
- State the source loss normalization and the source-only conditional Goji supervision. State the zero direct target GV loss.
- Use the main RGB26→RGB30 phy-only model for branch evidence. Full-model gains cannot be assigned solely to H-Phy.
- Separate new 15K experiments from historical 20K results. Teacher spatial maps and final Student IoUs have different identities.

`beta_gv=1`, minimum weight 0.75, initial gate strength 0.10 and full-strength point 4000 are engineering initial settings requiring experiments. The maximum relative attenuation is 25% on eligible pixels. The outputs represent learned VI-conditioned support. Quantitative accuracy and reliability improvements remain unmeasured until independent experiments finish.

## Research basis

The following sources were supplied in the task document. Their roles motivate the design, while the two-head decomposition and its coefficients are specific to this implementation.

- Tranheden et al., *DACS*, WACV 2021, motivates preserving cross-domain mixed sampling and self-training. [Official paper](https://openaccess.thecvf.com/content/WACV2021/html/Tranheden_DACS_Domain_Adaptation_via_Cross-Domain_Mixed_Sampling_WACV_2021_paper.html)
- Hoyer et al., *DAFormer*, CVPR 2022, motivates attention to class bias and trustworthy source supervision. [Official paper](https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html)
- Jiang et al., *Prototypical Contrast Adaptation*, ECCV 2022, motivates explicit class discrimination. Its prototype loss is separate from H-Phy. [Official paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4992_ECCV_2022_paper.php)
- Liu et al., *Validation of Red-Edge Vegetation Indices in Vegetation Classification in Tropical Monsoon Region*, Remote Sensing 2024, 16, 1865, motivates complementary vegetation indices across different vegetation conditions. [DOI](https://doi.org/10.3390/rs16111865)
- Liu et al., *A soybean mapping framework via multi-index-derived pseudo-label enhancement and contrastive embedding neural network*, ISPRS JPRS 2026, 235, 38–57, motivates joint analysis of multi-index discrimination and pseudo-label reliability. Its crop and implementation differ from this project. [DOI](https://doi.org/10.1016/j.isprsjprs.2026.02.039)

## Early nonzero 4K gate

Total training remains 15000 iterations. The gate starts at strength 0.10 at local_iter 0, reaches 1 at local_iter 4000, and stays at 1 through 15000. This schedule is independent of the inherited optimizer warmup and Poly learning rate. Training, online probes and replay use `phy_gate_strength` and `hierarchical_phy_gate` from `mmseg/models/uda/phy_reliability.py`.

Nine enabled leaves explicitly record `early_nonzero_linear`, initial strength 0.10, full iteration 4000 and gate floor 0.75. Fresh run identities use `hphy_early4k`. Resume/load checks reject a saved schedule mismatch. Replay restores historical 3000-iteration or delayed 6000/4000 timing from checkpoint metadata and records the difference.

The early small contribution and gradual introduction follow the training principles in Mean Teacher and Temporal Ensembling. The selected 0.10 and 4000 settings require experimental validation.

- Tarvainen and Valpola, *Mean teachers are better role models*, NeurIPS 2017. [Paper](https://arxiv.org/abs/1703.01780)
- Laine and Aila, *Temporal Ensembling for Semi-Supervised Learning*, ICLR 2017. [Paper](https://arxiv.org/abs/1610.02242)

Current acceptance evidence and the combined instruction checklist are recorded in `docs/hphy_early4k_acceptance.md`.
