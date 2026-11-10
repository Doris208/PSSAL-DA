# DORIS / PSSAL-DA Checkpoint 离线机制回放

## 本次交付与验证边界

只在本地修改、审计和测试代码；按用户要求不连接 SSH、不访问远端 GPU。
入口为 `tools/replay_doris_mechanisms.py`，复用现有 fixed_probe 的数据加载、模型分支计算和绘图函数。
不调用 `tools/train.py`、训练 runner、backward、optimizer.step、EMA 更新或原型更新。
构建模型时仅在内存中取消预训练加载，原始配置、Checkpoint 和训练结果均不改写。

**尚未完成真实 Full 模型两个历史节点的 GPU 回放。** 本地缺少可读取的目标实验 Checkpoint，且没有完整 MMCV/CUDA 运行环境。
两个合成 Checkpoint 的入口集成测试已覆盖加载、恢复、真实分支函数前向、MPARC/PBCR 绘图及独立空间图导出；使用小型替代 backbone，不能作为正式实验结果。
三个真实 TIF 的轻量测试确认输入为 `1×6×256×256`、VI 为 `1×3×256×256`，响应空间为 `256×256`。

## 本地 Checkpoint 清单（2026-11-09）

审计文件：`outputs/checkpoint_replay_audit_20261109/checkpoint_audit.json`。
逐文件记录路径、大小、SHA256、文件名 Iter 和读取错误。
`shared-nvme/work_dirs/doris_stage1` 下共发现 20 个 `.pth`，全部报 ZIP central directory 缺失，无法读取权重。
这只说明**本地副本不可用**，不代表训练服务器原文件损坏。

| 本地实验目录（省略 doris_stage1 前缀） | 文件名对应节点 |
| --- | --- |
| `goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge` | 仅 `latest.pth`，Iter 无法确认 |
| `goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge` | `iter_1000.pth`、`best_mIoU_iter_1000.pth`、`latest.pth` |
| `goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge` | `iter_1000.pth`、`best_mIoU_iter_1000.pth`、`latest.pth` |
| `goji26RGB2goji30RGB_hrda_3vi`（非 Full） | 16000、17000、18000、19000、20000、best 6000、latest |
| `goji26IRRG2goji30RGB_hrda_3vi`（非 Full） | 16000、17000、18000、19000、20000、best 11000 |

因此目前不能确认真实 Full 文件中学生、Teacher、原型及辅助缓存是否完整，也没有生成两个真实测试节点的机制图。其他实验的 Checkpoint 不用于替代。

## 固定样本与路径

工作目录默认取实验配置的 `work_dir`，图片路径由该方向配置的 `data.val.data_root / data.val.img_dir / 固定文件名` 解析。
同时验证训练目标域和 val 的域、视图一致；缺图报错，不自动换图，不读取标签。

| 方向 | 当前配置解析的绝对路径 |
| --- | --- |
| ZN_26 RGB → HN_30 RGB | `/root/shared-nvme/mydataset/30/rgb_dir/val/3_1024_1280_1280_1536.tif` |
| ZN_26 IRRG → HN_30 RGB | `/root/shared-nvme/mydataset/30/rgb_dir/val/3_1536_640_1792_896.tif` |
| HN_30 IRRG → ZN_26 IRRG | `/root/shared-nvme/mydataset/26/img_dir/val/3_768_964_1024_1220.tif` |

保留目标域六通道顺序、VI 读取与归一化；指定样本不进行随机裁剪、翻转或 Resize。三通道 `vi_img` 是生理分支输入，不会再次拼接进六通道 backbone 输入。
仅在显式使用本地镜像时传 `--target-data-root D:/Local_HRDA/shared-nvme/mydataset/30`（最后一级为目标域 `26` 或 `30`）。

## 状态恢复规则

| 项目 | 处理 |
| --- | --- |
| 学生和 EMA Teacher | 对照配置构建模型，逐键检查存在性、形状、有限值；审计阶段的 key 数量不等于完整性认证 |
| Teacher 缺失 | 不计算依赖其伪标签的机制图；不会用学生代替 |
| 生理头或结构先验缺失 | 停用对应响应及依赖它的下游权重；其余独立响应可以保留 |
| 融合原型和有效类别标记完整 | 使用已有 `compute_weight(..., read_only=True)`，不刷新融合原型 |
| 源／目标原型或计数缺失 | 即使融合原型可计算权重，也标记 `historical_state_complete=false` |
| 融合原型或学生缺失 | 原型响应不可用；不使用初始化原型补齐 |
| 邻域 patch 历史 | 若启用且未保存，边界分支标记不可恢复 |
| `feature_scale='all'` 依赖训练 crop | 无法恢复训练临时 crop，停用该原型响应 |
| 训练批次诊断 | ClassMix、随机 crop、逐批原型筛选／更新计数及训练专用结构特征交互不由此工具复现 |

必须存在 `checkpoint.meta.config`，并与提供配置的模型、分支和输入设置一致。日志与输出位置差异允许忽略；算法差异报错。
原始训练代码未保存完整源码哈希时，不能宣称逐位复现旧代码；输出记录当前实现的源码哈希。

编号文件的 `meta.iter` 允许为 N 或 N−1，后一种按 MMCV 的零基保存计数校正到完成 Iter N。
独立 `latest.pth` 只有指向编号文件或与编号文件字节完全相同时才可确定 Iter；不猜测。
同一 Iter 的模型状态冲突时不自动挑选。

计算使用 `eval()`、`no_grad()`；通过共享 fixed_probe 显式调用分支，避免误用仅返回分割结果的普通测试接口。
恢复模型模式、RNG 和 HRDA 推理临时状态，前后校验模型参数／buffer 哈希及 Checkpoint 哈希。
输出是“历史权重对固定图片的新前向”，不是原训练 batch 的临时张量。

## 先验证 RGB → RGB Full 两个节点

以下命令由用户在已配置好的原训练环境、仓库根目录执行。本次没有在远端执行。

```bash
CONFIG=configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py

# 自动使用 CONFIG 的 work_dir；先核查真实文件。
python tools/replay_doris_mechanisms.py "$CONFIG" --audit-only

# 从实际可读节点中选择：接近最终 Iter 一半的节点 + 最后节点。
# 少于两个不同的可读节点时明确失败。
python tools/replay_doris_mechanisms.py "$CONFIG" --pilot
```

也可指定 `--work-dir /实际实验目录`。如需手选节点，将 `--pilot` 改为
`--iterations 中期实际Iter 后期实际Iter`，数字必须来自审计结果。不要把日志中的验证 Iter 当成已保存的 Checkpoint。

验收两节点后再批量：核对 TIF 路径、Checkpoint 哈希和 Iter、`recovery` 恢复状态、原型类别有效性、图像原始尺寸、响应统计及前后差值。
缺失响应显示 N/A，部分恢复图显式写 PARTIAL；退出码 1 表示失败或部分恢复，需要查报告，不等于所有图片都没有输出。

## 三个方向批量命令

仅在小规模回放验收后执行：

```bash
for direction in goji26RGB2goji30RGB goji26IRRG2goji30RGB goji30IRRG2goji26IRRG; do
  config="configs/hrda/doris/${direction}/doris_${direction}_hrda_3vi_proto_struct_edge.py"
  python tools/replay_doris_mechanisms.py "$config" --all-checkpoints
done
```

每个方向只读取该配置工作目录的 Checkpoint，不递归混入其他实验。同一状态的 latest/best 别名不重复生成。

仅本地递归审计可使用 Torch 环境，不需要 MMCV/GPU：

```powershell
& D:\Anaconda3\envs\aglldiff\python.exe tools/replay_doris_mechanisms.py `
  --audit-only --recursive --work-dir shared-nvme/work_dirs/doris_stage1 `
  --out-dir outputs/checkpoint_replay_audit_20261109
```

## 输出与可追溯性

默认输出：`<work_dir>/analysis/fixed_probe/replayed/`，不覆盖 `snapshots/`。

```text
replayed/
  checkpoint_audit.json
  replay_report_<所选Iter>.json
  <固定TIF文件名去扩展名>/
    iter_<六位Iter>_<Checkpoint哈希前12位>/
      sample_manifest.json
      replay_metadata.json
      responses.npz
      iter_<六位Iter>_mparc.png
      iter_<六位Iter>_pbcr.png
      spatial/iter_<六位Iter>/*.png
      spatial/iter_<六位Iter>/metadata.json
```

NPZ、PNG Description 和 JSON 均标记 `checkpoint_replay`。记录 Checkpoint／恢复状态／配置／源码／TIF／输入张量哈希、Iter、分支状态、缺失键、统计及生成方式。
重复运行不覆盖不同来源或内容的结果；需要重新生成时可给出新的 `--out-dir`，例如 `<work_dir>/analysis/fixed_probe/replayed/review2`。
审计文件发生变化时同样需要新目录。

综合图复用现有大画布、300 dpi 输出；独立空间图与原图 H×W 严格一致，不增加坐标轴边距或重采样。
权重固定使用 [0,1]，差值固定 [-1,1]，边界监督强度固定 [0, 配置的目标域边界损失系数]；不逐图拉伸制造差异。

## 原型前后接近一致的检查

DACS 初始权重是整图高置信像素比例，因此有效区域可能是常数。原型门控处于 warmup、原型类别未就绪、上游权重为零等情况，均可能产生相同的前后权重。
该 Full 配置的原型权重从 Iter 6000 开始，`proto_min_weight=0.88` 也限制了后续权重变化幅度。差异小不能直接推断分支失效。

此前旧样本的已保存 NPZ 中，3000 节点处于原型 warmup、前后差值为零；6500 和 20000 节点最大绝对差值约为 0.046804 和 0.062613。
这些是旧样本证据，不能作为本次指定 TIF 的离线结果。
新回放会记录 `states`、`prototype.valid_class_ids`、`weight_start`、各响应 min/max/std、原型最大／平均绝对差值和变化像素比例，供逐节点判断；不会修改数值来放大区别。

## 本次文件范围

- 新增 `tools/replay_doris_mechanisms.py`：审计、节点选择、状态检查、只读回放、溯源输出。
- 修改 `mmseg/utils/mechanism.py`：抽取可复用的样本加载与分支数据打包函数，在线与回放共用原计算逻辑。
- 修改 `tools/build_doris_mechanism_figures.py`：回放标识、部分恢复标签和缺失响应兼容。
- 新增 `tests/test_doris_checkpoint_replay.py`：损坏文件、Iter、缺失状态、只读性、配置校验及两节点入口／绘图测试。
- 新增本说明文档。

本次没有修改网络、损失函数、实验配置或训练入口，也没有启动正式训练。

## phy 分支机制图与人工圈选

在仓库根目录的 `hrda` 环境执行 `python tools/replay_doris_phy_7000.py`。
同步代码时需包含 `tools/replay_doris_phy_7000.py`、`tools/replay_doris_mechanisms.py`、
`tools/doris_phy_evidence.py`、`tools/check_doris_phy_semantics_7000.py`、`tools/build_doris_mechanism_figures.py` 和 `mmseg/utils/mechanism.py`。
入口使用 RGB→RGB 的 `best_mIoU_iter_7000.pth` 和指定 TIF，输出到该实验的
`analysis/fixed_probe/replayed/phy_<Iter>_<月日_时分秒_微秒>/`，每次运行创建独立目录。
更换同一 phy 实验的历史权重可执行 `python tools/replay_doris_phy_7000.py --checkpoint iter_10000.pth`，
权重文件相对原 work-dir 解析，也支持绝对路径。Iter 从实际文件名及 Checkpoint 元数据核验获得。

论文综合图为单行五图，依次展示 RGB、EMA Teacher confidence、生理响应、权重差值和 GT 语义核对，标记为 (a)–(e)。
门控前后权重继续作为独立散图导出，其均值和门控前空间标准差写在综合图下方。
Teacher confidence 为逐像素最大类别概率。DACS 门控前权重采用整图高置信像素比例，两者在图题和元数据中分别定义。
图注围绕“VI 驱动的响应与权重调整，以及 Goji/其他植被的混淆核对”组织。
RGB 展示目标域场景，Teacher confidence 保留五类预测的逐像素最大概率，R_phy 保留生理头的真实输出。
当前生理头以 Goji 和其他植被共同作为正类，第三图明确标为 Learned VI-head response，类别区分相关的解读依据第五图及类内统计。

RGB 保持完整原始视图，虚框与重点区域由作者后续手动选取。辅助文件包括
`spatial/iter_007000/reference_label.png`（验证标注参考图）、`pseudo_label.png`（Teacher 预测图）和 `w_conf.png`。
类别颜色沿用 GojiDataset 的原调色板，忽略标签显示为灰色；原始类别 ID 保存在 `responses.npz` 中。
验证标签在模型前向完成后读取，使用原 `LoadAnnotations(reduce_zero_label=True)` 映射，按有效像素统计。

`phy_class_statistics.csv` 和同名 JSON 包含五类地物的有效像素数、Teacher 正确比例、响应／权重均值和标准差、减权比例。
图中缺失的类别记录像素数 0，统计值留空。`replay_metadata.json` 保存验证标注路径、哈希、标签映射、权重含义和生成方式。
局部区域的统计应依据作者最终选取的坐标另行计算；当前文件提供整图有效区域的类别统计。

### 论文展示版

同一条 `python tools/replay_doris_phy_7000.py` 命令会完整前向并导出论文版。
原始 256×256 张量和原图保留，新增文件位于当前节点目录的 `paper/`。
`iter_<六位Iter>_phys_paper.png` 与同名 PDF 使用共同的有效区域裁剪，本样本为 `[30:226,30:226]`，五幅子图均为中心 196×196 像素。
组图使用 `paper/spatial/`，其中所有空间图全部采用同一个裁剪范围，均为 196×196。
导出包括 `target_image.png`、`target_image.tif`、`w_conf.png`、`r_phy.png`、`w_before_phy.png`、
`w_after_phy.png`、`delta_w_phy.png`、`pseudo_label.png`、`reference_label.png`、`m_phy.png`、`valid_mask.png`、`phy_semantic_check.png`。
RGB TIF 是像素对齐的 RGB 展示裁剪图，PNG 和 TIF 像素一致。预测与参考标注沿用原类别调色板，掩码使用黑白表示。
`paper/spatial/metadata.json` 记录共同裁剪坐标与显示参数。原节点的 `spatial/iter_007000/` 保存全幅溯源图，人工组图使用 `paper/spatial/`。
RGB 保持干净，虚框由作者后续手动添加。

论文版缩短标题、增大字号、增加 (a)–(e) 面板标识，按各张图的物理含义设置色条。
置信度和生理响应使用 [0,1]。回放入口中前后权重散图共用 [0,1]，差值默认使用 [-0.10,0] 的线性蓝色色条。
其他 Checkpoint 的减权超出该范围时，入口按 0.05 的整倍数向外扩展下界，覆盖真实值。
显示尺度和裁剪范围写入 `iter_<六位Iter>_phys_paper_metadata.json`。跨节点直接比较时，应使用相同的显示范围。
实际值超出显示界限时程序会要求扩大范围。同一范围适用于需要直接比较的节点，具体界限可通过绘图入口的 `--weight-range`、`--delta-range` 指定。
图中注明有效区、门控前权重均值及空间标准差、门控后权重均值、平均差值、最大减权和变化像素比例，保留实际减权幅度的可读数值。

第五图核对 Goji 与其他植被之间的混淆，限定 GT 和 Teacher 均为 Goji/Vegetation 的有效像素。
四种颜色依次标出 Goji→Goji、Goji→其他植被、其他植被→Goji、其他植被→其他植被，箭头表示 GT→Teacher。
其他类别及 GT ignore=255 标为灰色。两种误判方向分别与同一 GT 类别中的判对像素比较，避免类别组成差异混入结论。
同一条回放命令自动生成 `paper/semantic_statistics.csv`、同名 JSON 和 `paper/semantic_summary.txt`。
综合图下方列出四组像素数、平均 R_phy 和平均 ΔW。语义核对空间图仅编码判别状态，减权大小由第四图及定量统计呈现。
CSV 主表对应这四组，JSON 的 `vegetation_class_check` 保存两个方向的差异和结论，原二大类统计保留为辅助信息。
结果允许双向支持、单向支持、缺组或未观察到预期方向。当前生理头将两类共同作为正类监督，
该核对描述权重调制与类间混淆的关系。区分性能的提升仍需对应的类别级消融指标支撑。

已有 NPZ 的本地排版检查也可单独执行 `python tools/build_doris_mechanism_figures.py /path/to/responses.npz --phy-paper`。
论文版文件与原图分目录保存。已有不同来源的论文版将报错，可通过 `--out-dir` 另选目录。

## RGB→RGB Baseline Iter 18000 对照入口

在同一 `hrda` 环境和项目根目录执行以下命令。

```bash
python tools/replay_doris_baseline_18000.py
```

默认读取 `/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB_hrda_3vi/iter_18000.pth`，
实验配置从该 Checkpoint 的 `meta.config` 恢复。入口核验四个机制分支均关闭，Student 和 EMA Teacher 完整恢复，
实际 Iter 与文件名和元数据一致。指定 TIF 仍为 `3_1024_1280_1280_1536.tif`，路径由历史数据配置的目标域 val 目录解析。
需要更换同一实验的节点时使用 `--checkpoint iter_20000.pth` 或权重绝对路径。

结果位于该权重所在目录的 `analysis/fixed_probe/replayed/baseline_18000_<时间戳>/`。
节点下的 `paper/iter_018000_baseline_paper.png` 和 PDF 为单行五图，
依次展示 RGB、EMA Teacher confidence、Teacher 预测、GT 参考和 Goji/其他植被混淆核对。
其中 confidence 范围为 [0,1]，类别色和混淆色与 phy 组一致，采用相同的 `[30:226,30:226]` 裁剪。
`paper/spatial/` 的全部散图和 RGB TIF 均为 196×196，节点 `spatial/iter_018000/` 保留原尺寸图。
`paper/semantic_statistics.csv` 和 JSON 保存四种类间判别状态的数量及 Teacher confidence 均值和标准差。
Baseline 的结果依据其实际前向输出生成，phy 组继续单独导出 R_phy、ΔW 和全部散图。

同步代码需包含新入口、`tools/replay_doris_mechanisms.py`、`tools/build_doris_mechanism_figures.py`、
`tools/doris_phy_evidence.py`、`tools/check_doris_phy_semantics_7000.py` 和 `mmseg/utils/mechanism.py`。

### DAFormer Baseline Iter 13000

```bash
python tools/replay_doris_daformer_13000.py
```

该入口复用 Baseline 回放与出图流程，默认读取
`/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB_daformer_3vi/best_mIoU_iter_13000.pth`。
模型从该权重的历史配置恢复，固定样本和 196×196 裁剪与 HRDA Baseline、phy 对照组一致。
结果位于 DAFormer 权重所在目录的 `analysis/fixed_probe/replayed/baseline_13000_<时间戳>/`，
综合图为节点下的 `paper/iter_013000_baseline_paper.png` 和同名 PDF，散图位于 `paper/spatial/`。
需同步 DAFormer 新入口、`tools/replay_doris_baseline_18000.py` 及支持 DAFormer 固定样本名称解析的 `mmseg/utils/mechanism.py`。

### DAFormer R101 无 VI Baseline Iter 15000

```bash
python tools/replay_doris_daformer_r101_no3vi_15000.py
```

默认读取 `/root/shared-nvme/work_dirs/doris_no3vi_r101/goji26RGB2goji30RGB_daformer_r101_no3vi/best_mIoU_iter_15000.pth`。
输入按历史配置使用三通道 RGB 及对应归一化，VI 加载器在未启用 VI 时直接返回，Collect 仅收集图像。
共享回放接口以零通道空张量表示 VI 缺席，样本元数据记录 `vi_available=false`，输入哈希对应真实 RGB 张量。
输出继续使用同一 TIF、一行五图、196×196 的论文散图、EMA Teacher confidence 和四组类间混淆统计。
结果位于该权重目录的 `analysis/fixed_probe/replayed/baseline_15000_<时间戳>/`，
节点下的 `paper/iter_015000_baseline_paper.png` 和 PDF 为综合图，`paper/spatial/` 保存散图与 RGB TIF。
