# H-Phy 机制图协议

## 训练在线与手动离线输出

RGB26→RGB30 的 Physiology、MPARC、Full 三组配置在每完成 500 Iter 后，使用当时内存中的模型，对六张固定目标图逐张以 B=1 只读前向。每图单独计算基础权重、H-Phy 响应、门控和统计，分别保存 NPZ、元数据和图像。探针恢复模块模式、随机数状态与推理临时属性，测试同时核对模型参数、梯度和优化器状态。

主样本沿用 `analysis/active_resume.json` 指向的记录目录，其余五图分别位于 `analysis/fixed_samples/<direction>__<图像名>_native_v1/`，各自的 `active_resume.json` 标记续接目录。六张文件名由三个配置中的 `fixed_probe_filenames` 明确指定。

每个样本的有效记录目录包含下列文件。

```text
manifest.json
fixed_probe/sample_manifest.json
fixed_probe/selected_arrays/iter_XXXXXX.npz
fixed_probe/snapshots/iter_XXXXXX_phys.png          # Physiology 四列组图
fixed_probe/snapshots/iter_XXXXXX_mparc.png         # MPARC / Full 对应组图
fixed_probe/snapshots/iter_XXXXXX_pbcr.png          # Full 的 PBCR 组图
fixed_probe/snapshots/full256/iter_XXXXXX/*.png
fixed_probe/snapshots/full256/iter_XXXXXX/metadata.json
fixed_probe/snapshots/center196/iter_XXXXXX/*.png
fixed_probe/snapshots/center196/iter_XXXXXX/metadata.json
```

Physiology 组图依次显示 (a) RGB、(b) Teacher confidence、(c) S_phy 和 (d) A_phy。b、c、d 的下方各有独立横向色标，confidence 与 S_phy 固定为 0～1，A_phy 固定为 0～1−phy_min_weight，当前为 0～0.25。其它 H-Phy 在线组图的连续响应也使用各自下置的横向色标。所有图片与迭代共享对应量的固定范围。

`full256` 保留完整网格及灰色无效边框，`center196` 对同一数组直接切片 `[30:226,30:226]`，裸图大小严格为 196×196。两份使用相同颜色映射，中心裸图与完整裸图对应区域逐像素一致。S_phy 仅显示有效 Teacher Goji/Vegetation 区域，A_phy/G_phy 进一步要求基础权重为正，其余区域为灰色。`attenuation_phy.png` 保存 A_phy，`g_phy.png` 单独保存 G_phy。

NPZ 和 metadata 保存实际 `rule_iter`、配置、模型状态 hash、样本身份和门控参数。A_phy 取实际训练调用链的 `1−G_phy`，当前 rho 从 Iter 0 的 0.10 线性增长到 Iter 4000 的 1，总训练长度仍为 15000 Iter。

更新代码需要重启并从 checkpoint 恢复训练进程。已有 Python 进程继续使用其加载的代码。原始历史文件保留，新加入的五张样本从更新后的采样节点开始积累响应。已有 NPZ 可用 `tools/build_doris_mechanism_figures.py --snapshots-only` 向独立输出目录重绘，复用保存数组。

本次在线调整涉及下列文件。

- `mmseg/utils/mechanism.py`
- `tools/build_doris_mechanism_figures.py`
- `tools/check_doris_stage1_configs.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py`
- `tests/test_hphy.py`
- `tests/test_hphy_online_panels.py`
- `docs/hphy_phy_figure_protocol.md`

上一轮在线实现回归为 194 PASS、1 SKIP，日志保存在 `outputs/hphy_online_full_regression_final.log`。唯一跳过项是缺少真实数据根目录的既有 TIFF 检查。24 组 DORIS 配置解析、三个方向九组 H-Phy 的 4K 调度检查、Python 编译和 `git diff --check` 均通过。新增测试使用 CPU 小型模型与合成输入，核查六图独立性、旧单图记录续接、重复节点复用、模型/梯度/优化器/RNG 状态、横向色标、显示作用域和精确裁剪。色标预览位于 `outputs/hphy_online_colorbars_synthetic_v2/iter_004000_phys.png`，图中明确标记合成数据。实际训练模型的 GPU 在线出图由作者上传后验证。

### 已开始训练时使用离线推理

训练进程已经启动时，可将更新后的代码上传，再用独立进程读取已保存的 checkpoint。离线 H-Phy 响应直接复用在线的 `mmseg.utils.mechanism.probe_arrays`，包括 EMA Teacher、softmax 精度路径、B=1 基础权重、有效 mask 和 `DACS._apply_phy_gate`。调度以 checkpoint 保存的实际参数和完成迭代为准，历史 3K/延迟调度仍按其原始配置回放。

离线默认逐张处理同样六个目标 TIFF。一次命令完成响应保存、五列论文图、独立子图、色条和在线兼容输出。五列图依次为 RGB、Teacher confidence、S_phy、A_phy、H-Phy Student 预测，b/c/d 下方均有横向色标。Teacher 类别图另存为 `panels_196/teacher_pseudo_label.png` 和对应 256 图，Baseline 预测与可选同位置 ROI 对照继续保留。

每个样本目录额外生成与在线探针同名的 `fixed_probe/selected_arrays/iter_XXXXXX.npz`、`fixed_probe/snapshots/iter_XXXXXX_phys.png`、`fixed_probe/snapshots/full256/iter_XXXXXX/` 和 `fixed_probe/snapshots/center196/iter_XXXXXX/`。这些文件由已保存数组生成，共用在线绘图器。在线兼容版和五列版的 RGB、confidence、S_phy、A_phy 裸图颜色与作用域一致，196 图均为同一 256 图的精确中心切片。

在项目根目录运行下列命令，H-Phy 使用指定编号权重，Baseline 默认使用其目录中的 `latest.pth`。

```bash
python tools/replay_hphy_mechanism.py --checkpoint iter_10000.pth
```

若两组目录均有 `iter_10000.pth`，可固定两份权重。

```bash
python tools/replay_hphy_mechanism.py --checkpoint iter_10000.pth --checkpoint-baseline iter_10000.pth
```

H-Phy 相对权重名位于 `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_gate/`，Baseline 相对权重名位于同级 `goji26RGB2goji30RGB_hrda_3vi/`。默认设备为 `cuda:0`。输出位于 `/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/hphy_mechanism_26rgb2_30rgb/<权重名>_<时间戳>/<图像名>/`，每次创建新目录。

本轮同步修改 `mmseg/utils/mechanism.py`、`tools/replay_hphy_mechanism.py`、`tools/build_hphy_mechanism_figure.py`、`tests/test_hphy_mechanism.py` 和本文。部署时同步上传共享的 `tools/build_doris_mechanism_figures.py`，以使用上一轮新增的 full256/center196 导出实现。

最终本地全仓回归为 **196 PASS、1 SKIP**，日志为 `outputs/hphy_sync_full_regression_final.log`。唯一跳过项仍为缺少真实数据根目录的既有 TIFF 检查。CPU 合成 checkpoint/TIFF 测试验证旧单图观测配置可以离线处理六图，同一权重和输入下公共响应数组及 dtype 一致，公共裸图与五列版散图 RGB 像素一致。含 ROI 和无 ROI 两种布局均检查标题边界及图例/色条/影像无重叠。CLI help、Bash 语法、Python 编译和 diff 检查通过。预览为 `outputs/hphy_sync_synthetic/phy_mechanism_1x5_final.png`，属于合成数据。真实训练权重与 GPU 的推理结果待作者上传后验证。

下文列出离线工具的完整接口和历史验收记录。

## 图的证据范围

本图展示 RGB26→RGB30 的 H-Phy 类别条件响应如何进入实际门控，并提供同一位置的 Baseline 与 H-Phy 预测对照。图版采用固定五列影像板，主响应为 `S_phy`，直接作用变量为 `A_phy=1-G_phy`。单图统计描述当前样本，性能与错误自强化效果需要独立实验支持。

Python/Matplotlib 负责所有组图、散图、色条和预览。组图宽 7.2 英寸，约 183 mm，默认 600 dpi、9 pt 字体，PDF 使用可选择的 TrueType 文字。所有空间影像保持等比例与 nearest 显示。

## 两个模型和真实调用链

仅支持下列两个 MiT-B5、六通道、五类、15K 配置。

- Baseline 使用 `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py`。
- H-Phy 使用 `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py`。

人工指定两个独立 checkpoint，可以采用各自的 best mIoU、best Goji IoU 或指定迭代。`--selection-phy` 和 `--selection-baseline` 可记录作者的选择依据。程序读取保存的训练配置，检查模块、输入、数据、种子、优化器、学习率、精度设置和 15K 协议。双头、Student、EMA 的完整 state_dict 必须匹配。历史单通道或缺少双头的权重会报错。

迭代来自编号文件与 `meta.iter` 的一致性检查，或明确保存的 `doris_checkpoint.completed_iter`。沿用项目的 MMCV 零基计数校验。两个 checkpoint 可有不同迭代，元数据分别记录其真实迭代和选择规则。该配对代表独立选择模型之间的比较。

正式验证的网络身份由当前 `UDADecorator.simple_test`、`single_gpu_test` 与 `tools/test.py` 核验为 Student。因此两份最终预测均通过 Student 的 `return_loss=False` 官方前向入口生成。(b) 使用 H-Phy 自身 checkpoint 的 EMA Teacher。CUDA 时按配置启用官方 AMP 包装，响应采集与在线探针共用 CUDA autocast。H-Phy 头的概率在 fp16 计算后转换为 float32 保存，门控计算为 fp32，CPU 路径使用 fp32。`checkpoint_phy.hphy_probability_dtype` 记录概率的计算精度，概率分解恒等式按该 dtype 的 epsilon 与 `1e-5` 中较大者校验，S 的类别选择、A/G 和权重恒等式仍按 `1e-5` 校验。

`eval()`、`torch.inference_mode()` 与现有 `readonly_probe` 共同约束回放。程序核对前后模型 state hash、梯度、checkpoint 文件 hash 和输入文件 hash。BN 状态、类别颜色和像素网格均有针对性测试。

## 输入和空间规则

默认目标图像列表在 Python 文件顶部的 TARGET_FILES 中定义，可通过 `--target-file` 指定一张或多张图覆盖。输入必须是 256×256、三波段 uint8 RGB TIFF，位于 `DATA_ROOT/rgb_dir/SPLIT/`。三个 VI 位于同一数据根目录的 `ndvi_dir/SPLIT/`、`ndre_dir/SPLIT/` 和 `cire_dir/SPLIT/`，文件名与目标 TIFF 相同。此约定来自现有 GojiDataset 的索引定位逻辑。

默认保留 3_1024_1280_1280_1536.tif，并新增 3_256_1152_512_1408.tif、3_1536_512_1792_768.tif、3_1536_1024_1792_1280.tif、3_1792_1280_2048_1536.tif、3_1536_640_1792_896.tif，六张图均位于同一 val 文件夹。每张以 B=1 独立前向，分别计算 DACS 基础权重、门控、响应统计和描述性指标。每图的 NPZ、JSON、1×5 组图、独立子图与色条保存在本次运行目录下以图像名命名的子目录中。

跨图输出保持独立，S 与 A 沿用固定色标。显式指定一张 --target-file 时保持原来的单图目录结构。--gt-file 对应单图参考标注，多图时应分别指定图像和其对应标注执行，避免错误配对。

读取和 Normalize 复用 GojiDataset、LoadGojiIndices 和现有目标域通道顺序。六通道输入为 RGB、NDVI、NDRE、CIRE，H-Phy 独立输入为归一化三 VI。程序核对两模型得到相同张量，并将反归一化的前三通道与原始 TIFF RGB 像素交叉核对。

网络前向全部在完整 256×256 网格完成。中心图只做数组切片 `[30:226,30:226]`，得到 196×196。ROI 使用原始网格的 `[x1,y1,x2,y2]` 坐标，上界不包含。ROI 必须完整落在中心显示区 `[30,30,226,226]`，中心坐标减去 30。越界和反向范围会报错。

所有 clean 图均直接编码当前数组对应的原始像素网格。较高 DPI 只影响带标签图和组图的排版像素数。五列等宽，预测 ROI 框位于 H-Phy 全幅预测上，两个同位置、同尺寸的 ROI 放在该列下方。默认关闭 confidence inset，单独输出 confidence 子图。

## 实际门控与权重

Teacher 来自已恢复的 EMA，H-Phy 响应调用 `DACS._apply_phy_gate`，进而调用 `hierarchical_phy_gate`。当前实现的分解和连续门控为

```text
P_G = R_veg*Q_G
P_V = R_veg*(1-Q_G)
P_N = 1-R_veg
S_phy = P_G for Teacher 0, P_V for Teacher 1, neutral 1 for other predictions
rho = 0.10 + 0.90*clip(completed_iter/phy_gate_full_iter,0,1)
phy_gate_full_iter = 4000, total training = 15000
G_phy = 1-rho*(1-g_min)*(1-S_phy) on spatial-valid Teacher 0/1
G_phy = 1 elsewhere
A_phy = 1-G_phy
W_after = W_before*G_phy
```

参数读取实际配置与模型属性。图中 S 和 A 的非适用区域使用灰色。NPZ 保留计算中的中性值和独立 mask，统计只取有效作用域，灰区不会作为零响应进入均值。

DACS 基础权重是整个 batch 的高置信像素比例乘有效 mask。单幅离线回放按同一公式在 `B=1` 的完整目标图上重算 `W_before`。历史训练 batch 的原标量无法由单样本 checkpoint 恢复，元数据明确记录这一边界。逐像素 `teacher_confidence` 单独保存。

`S_phy` 和 confidence 色标固定为 `[0,1]`，A 默认使用 `[0,1-g_min]`，当前为 `[0,0.25]`，ramp-up 阶段也采用共同理论范围。`--attenuation-vmax` 可覆盖显示范围并记录到 `render_meta.json`，跨样本比较应使用同一范围。概率、响应和权重数值保持原值。

## GT 和少量描述统计

GT 可选，且在两个模型前向完成后读取。默认接收项目原始 GT，复用 `LoadAnnotations(reduce_zero_label=True)` 得到 0 Goji、1 Vegetation、2 Building、3 Water、4 Road、255 Ignore。

`phy_stats.json/csv` 仅统计中心 196 和有效、非 ignore 像素。四组为 GT→Teacher 的 Goji→Goji、Goji→Vegetation、Vegetation→Vegetation、Vegetation→Goji，分别记录数量、S、A、confidence 的均值、总体标准差及 JSON 分位数。高置信错误阈值取实际 DACS `pseudo_threshold`。

同图 AUROC 的正类为两种植被类别之间的 Teacher 错误，评分采用 A，与 `1-confidence` 对照。少于 10 个有效像素或缺少正确/错误组时返回 null 并写明原因。没有 GT 时，组图、散图和数组仍可生成，GT 统计记录缺失。

## 输出文件

```text
RUN_DIR/
  responses_full_256.npz
  responses_center_196.npz
  mechanism_meta.json
  phy_mechanism_meta.json
  render_meta.json
  phy_stats.json
  phy_stats.csv
  phy_mechanism_1x5.png
  phy_mechanism_1x5.pdf
  panels_196/
    a_target.png  b_teacher.png  c_s_phy.png  d_attenuation.png  e_prediction.png
    teacher_confidence.png  baseline_prediction.png  hphy_prediction.png
    ndvi.png  ndre.png  cire.png  R_veg.png  Q_G.png  P_G.png  P_V.png  P_N.png
    G_phy.png  W_before.png  W_after.png
    baseline_vs_hphy_roi.png       # 指定 ROI 时
    clean/                        # 上述空间图的逐像素裸图
    labeled/                      # 简短标签与适用色条
  panels_256/                     # 对应完整网格散图、clean、labeled 和可选 ROI 对照
  colorbars/s_phy.png
  colorbars/attenuation.png
  colorbars/confidence.png
  logs/replay.log
  logs/render.log
```

主 NPZ 包括请求中的所有必需字段。`image_rgb` 为原始 uint8 RGB，`image_6ch` 为 HWC 形式的归一化六通道输入。`ndvi/ndre/cire` 为六通道输入中的归一化前指数值，`vi_ndvi/vi_ndre/vi_cire` 为实际 H-Phy 头输入。概率、门控和权重使用 float32，类别使用原索引，mask 单独保存。当前无需额外 TIFF 导出即可保留无损数值。

`mechanism_meta.json` 保存输入和两个 checkpoint 的路径、hash、迭代、模型身份、归一化、类别、crop、ROI 坐标约定、理论色标、软件版本和源码 hash。`render_meta.json` 继承推理 provenance，并记录该次重绘实际 ROI、颜色范围、字体、DPI 和渲染代码 hash。它是推理元数据中 `render_settings_file` 指向的显示参数记录。输入 NPZ 和原推理元数据均参与重绘前后 hash 检查。

默认使用全新运行目录。已有图、统计、目录或日志造成明确失败。前向入口会在响应保存后自动绘图，全部结果写入同一运行目录。独立渲染器继续支持从已保存数组重绘。已有 `doris_stage1` 路径被输出检查拒绝。

## 三种运行方式

以下独立 Python 示例在上传后的项目根目录执行，环境变量指向实际权重、RGB TIFF 和全新分析目录。Shell 读取 Python 文件顶部的路径默认值。

### 一条命令完成推理与绘图

默认配置、两组训练目录、checkpoint 文件名、目标 TIFF 和输出根目录集中定义在 `tools/replay_hphy_mechanism.py` 顶部。相对 checkpoint 名从对应训练目录查找，绝对路径直接使用。默认两组均读取 latest.pth，每次自动创建带 checkpoint 名和时间戳的输出子目录。

```bash
python tools/replay_hphy_mechanism.py --checkpoint iter_10000.pth
```

也可通过下列完整参数覆盖默认值。

```bash
python tools/replay_hphy_mechanism.py \
  --config-phy configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py \
  --checkpoint-phy "$CHECKPOINT_PHY" \
  --config-baseline configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py \
  --checkpoint-baseline "$CHECKPOINT_BASELINE" \
  --target-file "$TARGET_FILE" --out-dir "$RUN_DIR" --device cuda:0
```

可选增加 `--gt-file "$GT_FILE"`、`--roi 80 80 145 145`，以及两项 `--selection-*` 描述。程序根据权重元数据恢复迭代。

### 只从现有 NPZ 重绘

```bash
python tools/build_hphy_mechanism_figure.py \
  --input-npz "$RUN_DIR/responses_full_256.npz" \
  --meta-json "$RUN_DIR/mechanism_meta.json" \
  --output-dir "${RUN_DIR}_redraw" --render-only \
  --crop-mode center196 --export-group --export-panels --export-full256 \
  --roi 80 80 145 145 --font-size 9 --dpi 600
```

此入口只有 NumPy/Matplotlib 依赖，不导入 Torch，适合 CPU 重绘。

### Shell 一键回放并绘图

```bash
bash scripts/run_hphy_mechanism_26rgb2_30rgb.sh
```

Shell 采用一段 echo 与一条简短 Python 命令的平铺形式，读取 Python 文件顶部的默认值。replay_hphy_mechanism.py 自动依次保存响应并绘制 1×5、独立子图和色条。前向入口支持 --dpi、--font-size 和 --roi，回放和渲染各自记录日志。

## 原离线实现阶段的合成示例和验收

本节记录调度调整前的离线实现验收。当前 4K 调度的逐项结果见 [合并验收报告](D:/PSSAL-DA/docs/hphy_early4k_acceptance.md)。

示例目录为 `outputs/hphy_mechanism_synthetic_v2/final_sample/`，包含 600 dpi 组图、PDF、完整散图和数值文件。示例全部是 synthetic 布局测试。组图和带标签图具有明显标识，裸图保留 synthetic PNG metadata。示例不能作为论文机制结果。

测试使用合成数组、实际 TIFF/Goji 读取流程、两个小型 PyTorch synthetic checkpoint，以及真实 H-Phy 头和门控。小模型构建用测试替身明确隔离，并标记 synthetic。实际训练 checkpoint、真实目标样本和 GPU 在本次本地任务中不可用，因此真实已训练模型回放仍待作者上传权重与样本后执行。

该阶段完整回归结果为 **180 PASS、1 SKIP**，耗时 179.10 秒。SKIP 为已有真实 TIF 检查，原因是未配置实际数据根目录。

- 全仓日志为 `outputs/hphy_mechanism_final_regression.log`。
- 第一轮新测试为 13 PASS，最终又增加了 renderer 经 tee 失败的专项测试，并纳入上述全仓回归。
- 实际 Shell render-only 调用日志为 `outputs/hphy_mechanism_shell_real.log`，使用合成 NPZ 成功生成全部输出。
- Python py_compile、Bash bash -n/help、git diff --check 通过。PDF 面板文字可提取，600 dpi 示例图已查看。
- 示例共 128 个文件，完整相对路径保存在示例 `render_meta.json` 的 `output_files`。

本轮文件清单如下。已有 H-Phy 正式训练代码属于上一轮工作树内容。

- `tools/replay_hphy_mechanism.py`
- `tools/build_hphy_mechanism_figure.py`
- `scripts/run_hphy_mechanism_26rgb2_30rgb.sh`
- `tests/test_hphy_mechanism.py`
- `tests/test_hphy_mechanism_shell.py`
- `docs/hphy_phy_figure_protocol.md`
- `.gitattributes` 增加新 Shell 的 LF 规则

本轮独立补丁为 `outputs/hphy_mechanism_changes.patch`，基于任务开始时已经存在的 H-Phy 工作树。

## 英文 caption 模板

*Class-conditional physiological agreement and supervision modulation of H-Phy in the ZN_26 (RGB) → HN_30 (RGB) setting. Panels show (a) the target image, (b) EMA-teacher confidence, (c) physiological agreement, (d) relative attenuation and (e) H-Phy Student predictions. An optional paired ROI includes the matched Baseline Student comparison using identical original-image coordinates. All spatial maps display the central 196 × 196 pixels of the same 256 × 256 target patch. Checkpoint iterations and selection criteria are reported separately.*

## 参考定位

该五列模板及变量组合来自本项目的证据需求。以下公开来源提供自训练、类间判别、VI 分析和图表排版原则，引用定位沿用任务提供的信息。

- DACS，WACV 2021，支撑混合自训练和伪监督关系的说明。[论文](https://openaccess.thecvf.com/content/WACV2021/papers/Tranheden_DACS_Domain_Adaptation_via_Cross-Domain_Mixed_Sampling_WACV_2021_paper.pdf)
- DAFormer，CVPR 2022，支撑 Teacher 最大概率与训练权重的区分。[论文](https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html)
- ProCA，ECCV 2022，提供显式类别判别的研究动机。[论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4992_ECCV_2022_paper.php)
- Red-edge VI validation，Remote Sensing 2024，提供植被指数差异分析的动机。[DOI](https://doi.org/10.3390/rs16111865)
- Soybean multi-index mapping，ISPRS JPRS 2026，提供多指数与伪标签可靠性关联评价的动机。[DOI](https://doi.org/10.1016/j.isprsjprs.2026.02.039)
- IEEE Proceedings 图表指南，提供简洁标签、统一字体和导出质量原则。[指南](https://proceedingsoftheieee.ieee.org/resources/guidelines-for-figures-and-tables/)

## 4000 Iter 同步验收

当前门控以 Iter 0 的 10% 强度开始，Iter 4000 达到完整强度，总训练长度为 15000 Iter。`DACS._apply_phy_gate` 与离线 replay 共同调用 `hierarchical_phy_gate`，其中 `phy_gate_strength` 读取模型的实际配置。纯 NumPy/Matplotlib 渲染器读取数组和保存的 rho，校验 A 的门控恒等式。

历史 H-Phy checkpoint 按其保存的调度回放。`checkpoint_phy` 元数据分别记录请求参数、实际恢复参数及历史调度标识。新 4K 训练会拒绝恢复 3K 或延迟调度的 checkpoint。

迭代优先取明确的 completed_iter 和规范编号文件。普通别名会尝试核对 `checkpoint_best.json` 的文件身份或字节完全相同的编号权重。证据不足时可传 `--iter N` 和 `--iter-baseline N`，与已有可靠信息冲突会报错。配置中的总训练步数只用于协议边界检查。

`phy_mechanism_meta.json` 与兼容文件 `mechanism_meta.json` 保存相同 provenance。新增字段包括 gate_schedule、gate_initial_strength、gate_full_iter、gate_rho_at_checkpoint、gate_min_weight、gate_active_pixel_ratio、mean_S_phy_on_active_pixels、mean_A_phy_on_active_pixels 和 max_A_phy_on_active_pixels。无实际作用像素时统计为 null，并写 gate_empty_reason。

NPZ 独立保存 teacher_phy_mask、positive_base_weight_mask、positive_phy_weight_mask 和 gate_active_mask。S 显示有效 Teacher 植被预测域，A/G 显示具有正基础监督权重的 Teacher 植被域。可选 GT ignore 只参与离线描述统计。1×5、散图、固定色标与 256→196 中心裁剪均沿用既定协议。
