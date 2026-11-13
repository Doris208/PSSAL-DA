# 26IRRG→30RGB DORIS V2 实施与验收

基线提交为 `1377b878049d8d37edafdf4ed339ed1540e2169f`，本地分支为 `doris`。开始实施时 HEAD 与参考提交一致，工作区干净。

## A. 修改文件

配置目录 `configs/hrda/doris/goji26IRRG2goji30RGB/` 的八份原文件均已直接更新。

| 文件后缀 | 实验 |
|---|---|
| `hrda_3vi.py` | Baseline |
| `hrda_3vi_phy_gate.py` | H-Phy |
| `hrda_3vi_struct_gate.py` | Structural V2 |
| `hrda_3vi_phy_struct_gate.py` | MPARC |
| `hrda_3vi_ps_proto.py` | Prototype |
| `hrda_3vi_struct_edge.py` | Edge |
| `hrda_3vi_pbcr.py` | PBCR |
| `hrda_3vi_proto_struct_edge.py` | Full |

其余文件按用途列出。

- `.gitattributes` 为新脚本和总脚本固定 LF。
- `scripts/run_doris_tableiv_26irrg_to_30rgb_hphy15k.sh` 为新增独立八组脚本。
- `train_doris_mitb5_stage1_ablations_20k.sh` 更新 IRRG26 对应八条输出路径。
- `tools/check_doris_stage1_ablation_configs.py`、`tools/check_doris_hphy_configs.py`、`tools/check_doris_stage1_configs.py` 扩展双方向 V2 审计和参数一致性检查。
- `mmseg/utils/mechanism.py` 补齐真实组合权重、边界分量、原型权重及状态、结构证据和调度元数据。
- `tools/doris_probe_data.py` 增加源域视图与目标域身份交叉验证。
- `tools/replay_hphy_mechanism.py`、`tools/replay_struct_mechanism.py` 扩展成对回放的方向、配置、路径及元数据解析。
- `tools/replay_baseline_prediction.py` 支持显式 Baseline Checkpoint 和真实 Iter，增加方向选择。
- `tools/build_hphy_mechanism_figure.py`、`tools/build_struct_mechanism_figure.py` 扩展支持的方向。
- `tools/build_branch_prediction_comparison.py` 保存方向、实际 Iter 和训练步数是否匹配。
- `tools/replay_doris_mechanisms.py` 将新增的源域/目标域通道日志名称作为观察元数据处理。模型、实际数据配置、归一化和分支计算配置继续参与校验。
- `tools/check_doris_irrg_inputs.py` 为新增真实文件和生产加载流水线预检工具。
- `tools/smoke_doris_hphy.py` 增加方向、源 EMA/双路径调度检查、Checkpoint 空间检查入口和可选真实固定探针初始化。
- 新增 `tests/test_irrg_rgb30_upgrade.py`、`tests/test_irrg_probe_cpu.py`。
- 更新 `tests/test_baseline_prediction.py`、`tests/test_baseline_prediction_cpu.py`、`tests/test_doris_branch_outputs.py`、`tests/test_doris_checkpoint_policy.py`、`tests/test_doris_stage1.py`、`tests/test_hphy.py`、`tests/test_hphy_launcher.py`、`tests/test_struct_mechanism.py`，使断言对应本轮协议，并保留历史模式及其它方向回归检查。
- 本验收报告 `docs/irrg26_rgb30_v2_acceptance_zh.md`。

## B. 八组与最新 RGB 方向逐项比较

| 实验 | Phy/Struct/Proto/Edge | 与 RGB26 对应组的模块参数 | 方向固有差异 |
|---|---|---|---|
| Baseline | 0/0/0/0 | 相同 HRDA 基线 | IRRG 源域数据、归一化、通道日志及方向身份/输出目录 |
| H-Phy | 1/0/0/0 | hierarchical_gv、Early-4K 及全部 Phy 参数一致 | 同上，六探针首图按 IRRG 方向排序 |
| Structural | 0/1/0/0 | V2 四项证据及 Early-4K 参数一致 | IRRG 源域协议、方向身份/输出目录及指定首图 |
| MPARC | 1/1/0/0 | 与两个 only 分支一致 | IRRG 源域协议、方向身份/输出目录及六探针顺序 |
| Prototype | 0/0/1/0 | 3000/6000/6000、0.001、0.88、0.7 等参数一致 | IRRG 源域协议、方向身份/输出目录及指定首图 |
| Edge | 0/0/0/1 | balanced loss 与 0.01/0.006 等参数一致 | IRRG 源域协议、方向身份/输出目录及指定首图，保留历史无效字段 |
| PBCR | 0/0/1/1 | Prototype 与 Edge 参数均一致 | 同 Edge |
| Full | 1/1/1/1 | 四模块分别与对应 only 分支一致 | IRRG 源域协议、方向身份/输出目录及六探针顺序，保留历史无效字段 |

`boundary_pair_min_weight=0.80` 保留在 Edge/PBCR/Full。生产路径 `doris_reliable + valid_boundary_only` 直接读取有效边界条件响应，绕过历史 pair gate 的 minimum-weight 映射。CPU 测试将对应运行属性分别设为 0.01 和 0.99，所有实际边界响应逐元素一致。

Structural V2 的生产代码版本名仍为 `struct_distinct_v1`，与参考方向一致。H-Phy、Structural、Prototype、Edge 的计算实现通过参考提交内容回归检查。MiT-B5、HRDA、6 通道、256×256、batch=2、15K、seed=0、AdamW 6e-5、poly、DACS/EMA、ClassMix、0.968 阈值及五类映射在配置测试中检查。

## C. 两份训练脚本

独立脚本有八条独立命令，总脚本保留二十四条命令。配置审计核对顺序、配置路径和绝对输出目录。参考提交对比核验其它两个方向的命令内容。两份脚本的 LF 字节检查与 `bash -n` 均通过。

两个方向脚本各自通过五项假训练测试，覆盖默认顺序、独立目录、重复目录冲突、复制单条命令、失败即停、无效参数和缺失配置处理。测试在隔离临时目录使用替身训练命令。

## D. 六通道核验

| 项目 | 源域 | 目标域 |
|---|---|---|
| 数据根目录 | `/root/shared-nvme/mydataset/26` | `/root/shared-nvme/mydataset/30` |
| 图像目录 | `img_dir/train` | `rgb_dir/train`、`rgb_dir/val`、`rgb_dir/test` |
| 流水线后的通道 | NIR、R、G、NDVI、NDRE、CIRE | R、G、B、NDVI、NDRE、CIRE |
| mean | 111.366、122.724、135.261、0、0、0 | 144.074、140.700、133.597、0、0、0 |
| std | 61.302、62.369、55.989、1、1、1 | 63.564、65.998、65.842、1、1、1 |
| channel_order | `[2,1,0,3,4,5]` | `[2,1,0,3,4,5]` |
| to_rgb | False | False |

默认 OpenCV 解码将存储的三个波段反序，`LoadGojiIndices` 附加三个 VI，`Normalize.channel_order` 再交换前三通道。最终前三通道恢复存储顺序。分割输入使用原有 VI 输入处理，独立生理输入使用已有 range 归一化。

本地合成测试执行生产 `LoadImageFromFile`、`LoadGojiIndices`、`Normalize` 的方法，使用模拟 BGR 解码核对六个哨兵波段及独立 VI 的次序和数值。Baseline 新增的源域/目标域通道日志与其实际流水线相符。

真实传感器波段身份沿用项目数据契约。当前本地 `shared-nvme/mydataset` 中文件数为零，真实文件对应的六通道张量核验待服务器执行下列预检。工具逐项检查源域一张样本、目标六张图块及 VI，执行真实生产 Dataset 加载，反归一化比较存储波段，并记录文件和张量 SHA256。

```bash
python tools/check_doris_irrg_inputs.py --output /root/shared-nvme/irrg26_rgb30_input_check.json
```

## E. 六样本固定探针

目标图像根目录为 `/root/shared-nvme/mydataset/30/rgb_dir/val/`。

1. `3_1536_640_1792_896.tif`
2. `3_1024_1280_1280_1536.tif`
3. `3_256_1152_512_1408.tif`
4. `3_1536_512_1792_768.tif`
5. `3_1536_1024_1792_1280.tif`
6. `3_1792_1280_2048_1536.tif`

每个同名图块还需要位于 `30/ndvi_dir/val/`、`30/ndre_dir/val/`、`30/cire_dir/val/` 的三个 VI 文件。H-Phy/MPARC/Full 使用以上六样本顺序，Structural/Prototype/Edge/PBCR 在线使用第一张样本。离线成对回放默认支持全部六张。

样本名、数量、集合及首图已通过配置检查。实际文件存在性为待验收项，原因是本地数据目录为空。生产探针与新增预检均会对缺失文件报错。

## F. 回放与比较

- H-Phy 与 Structural 的 `--direction goji26IRRG2goji30RGB` 会选择对应配置及 Baseline 配置。显式绝对 Checkpoint 路径直接使用，方向和真实源/目标数据身份交叉验证。
- H-Phy 保留层次化计算、历史门控调度恢复及标准 1×5，新增完整 Student 测试设置元数据。
- Structural 保留 Checkpoint 内源邻接/语义概率原型 EMA，调用 `update_priors=False`。V2 按四项可用证据公式检查，保存原始尺度、availability、贡献和衰减。标准 1×7 的最终预测采用 Student 测试入口。
- Baseline 默认入口仍选择 RGB26 的 Iter 4000。新方向要求 CLI 提供真实 Checkpoint，实际 Iter 从 Checkpoint 元数据与文件/已验证记录交叉确认，支持 15K 内不同 Iter/Best 文件。
- 通用回放复用已有状态恢复，补齐 Prototype、Edge、PBCR、MPARC、Full 的固定输入响应字段和模块组合中间权重。输出保留 `capture_mode=checkpoint_replay`，元数据列明训练批次随机状态、ClassMix 及训练期特征交互的恢复范围。
- `tools/doris_struct_evidence.py`、`tools/struct_mechanism_panels.py`、`tools/build_doris_mechanism_figures.py` 现有实现已支持记录的融合版本和方向无关绘图，本轮通过检查与回归测试复用。
- 比较工具校验同一图块、RGB/VI 哈希、真实六通道数组、空间有效区域、中心裁剪、调色板和已记录 Student 推理设置，图中标注每个模型的实际 Iter。提供 GT 时导出分类别 IoU、Goji/Vegetation 混淆、新增正确及新增错误像素统计。
- 缺失历史模块状态沿用原有部分恢复标记。新增字段仅在对应分支实际执行时导出。旧 V1 配置可按历史协议回放，CPU 测试确认 V1 Checkpoint 配置与请求的 V2 配置发生冲突时拒绝加载。

通用分支的 Full 回放命令如下。相同入口也接受 Prototype、Edge、PBCR 和 MPARC 的配置及其独立工作目录。

```bash
python tools/replay_doris_mechanisms.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py --iterations 15000 --target-filenames 3_1536_640_1792_896.tif --out-dir /root/shared-nvme/irrg26_rgb30_replay/full
```

以下命令用于真实 15K Checkpoint 产生后的单样本验收。按需要将三个 Checkpoint 路径中的 Iter 文件替换为实际产物。

```bash
python tools/replay_baseline_prediction.py --direction goji26IRRG2goji30RGB --checkpoint /root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi/iter_15000.pth --target-file /root/shared-nvme/mydataset/30/rgb_dir/val/3_1536_640_1792_896.tif --out-dir /root/shared-nvme/irrg26_rgb30_replay/baseline

python tools/replay_hphy_mechanism.py --direction goji26IRRG2goji30RGB --checkpoint /root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_phy_gate/iter_15000.pth --checkpoint-baseline /root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi/iter_15000.pth --target-file /root/shared-nvme/mydataset/30/rgb_dir/val/3_1536_640_1792_896.tif --out-dir /root/shared-nvme/irrg26_rgb30_replay/hphy

python tools/replay_struct_mechanism.py --direction goji26IRRG2goji30RGB --checkpoint /root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_struct_gate/iter_15000.pth --checkpoint-baseline /root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi/iter_15000.pth --target-file /root/shared-nvme/mydataset/30/rgb_dir/val/3_1536_640_1792_896.tif --out-dir /root/shared-nvme/irrg26_rgb30_replay/struct

python tools/build_branch_prediction_comparison.py --baseline-npz /root/shared-nvme/irrg26_rgb30_replay/baseline/3_1536_640_1792_896/baseline_prediction.npz --phy-npz /root/shared-nvme/irrg26_rgb30_replay/hphy/responses_full_256.npz --struct-npz /root/shared-nvme/irrg26_rgb30_replay/struct/responses_full_256.npz --out-dir /root/shared-nvme/irrg26_rgb30_replay/comparison
```

## G. 本地测试结果

按唯一用例汇总，当前验证覆盖 **175 PASS、1 SKIP、0 个待修复 FAIL**。唯一跳过项需要真实 TIF 数据。

| 检查 | 实际结果 | 记录 |
|---|---|---|
| 真实 MMCV 24 组配置及 24 条总脚本命令 | PASS 24/24 | `outputs/irrg_upgrade_mmcv_audit.log` |
| H-Phy 配置检查器的新方向入口及跨方向检查 | PASS | `outputs/irrg_upgrade_hphy_audit.log` |
| CPU、输入、在线/离线回放、历史兼容、绘图完整联合运行 | 163 PASS、1 FAIL、2 SKIP，失败及其中一个跳过项随后补测 | `outputs/irrg_upgrade_joint_tests.log` |
| 最终配置/CLI/Baseline CPU/测试兼容定向复测 | 11 PASS | `outputs/irrg_upgrade_final_checks.log` |
| 两个方向的批量脚本及旧 Structural 回放脚本 | 11 PASS | `outputs/irrg_upgrade_launcher_tests.log` |
| 两份训练脚本的 Bash 语法与 LF | PASS | `bash -n` 与文件字节检查 |
| 修改 Python 文件的 Python 3.8 语法、Git 空白检查 | PASS | `ast.parse(feature_version=(3,8))`、`git diff --check` |

完整联合运行中的唯一失败来自原测试使用 `ast.unparse`，该 API 在 Python 3.8 中缺失。已改用相同语义的 AST 结构匹配，相关测试包含在最终 11 项定向复测中并通过。联合运行中因 Bash 路径未配置而跳过的旧 Structural 脚本测试，也在最后 11 项脚本测试中通过。真实 TIF 检查仍标记 SKIP。

本地完整联合测试的复现命令如下。

```powershell
$env:TMP='D:\PSSAL-DA\outputs\irrg_test_tmp'
$env:TEMP=$env:TMP
$env:PYTHONPATH='D:\PSSAL-DA\outputs\hphy_deps;D:\PSSAL-DA\tests'
$env:DORIS_PLOT_PYTHON='D:\Anaconda3\envs\aglldiff\python.exe'
& 'D:\Anaconda3\envs\aglldiff\python.exe' -m pytest tests/test_irrg_rgb30_upgrade.py tests/test_irrg_probe_cpu.py tests/test_baseline_prediction.py tests/test_baseline_prediction_cpu.py tests/test_struct_probe_cpu.py tests/test_struct_distinct_v1.py tests/test_hphy.py tests/test_hphy_mechanism.py tests/test_struct_mechanism.py tests/test_struct_visualization.py tests/test_hphy_online_panels.py tests/test_doris_fixed_probe.py tests/test_doris_checkpoint_policy.py tests/test_doris_stage1.py tests/test_doris_checkpoint_replay.py tests/test_doris_mechanism.py tests/test_doris_branch_outputs.py -q -rs --disable-warnings
```

最终定向复测覆盖以下命令。

```powershell
& 'D:\Anaconda3\envs\aglldiff\python.exe' -m pytest tests/test_irrg_rgb30_upgrade.py::ConfigUpgradeTests tests/test_baseline_prediction.py::BaselinePredictionTests::test_baseline_config_preprocessing_and_cli tests/test_hphy_mechanism.py::PairedReplayTests::test_short_checkpoint_command_uses_defaults_and_separate_output_directories tests/test_struct_mechanism.py::StructuralLauncherTests::test_resolved_configs_pass_shared_protocol_and_match_training_data tests/test_baseline_prediction_cpu.py -q -rs --disable-warnings
```

两方向批量脚本与旧回放脚本使用隔离替身命令完成测试。

```powershell
$env:PATH='D:\Git\bin;' + $env:PATH
$env:HPHY_BASH='D:\Git\bin\bash.exe'
& 'D:\Anaconda3\envs\aglldiff\python.exe' -m pytest tests/test_hphy_launcher.py tests/test_struct_mechanism.py::StructuralLauncherTests::test_launcher_failure_propagates_and_arguments_are_rejected -q
```


在具备原训练依赖的环境执行正式配置解析。

```bash
python tools/check_doris_stage1_ablation_configs.py
python tools/check_doris_hphy_configs.py --direction goji26IRRG2goji30RGB --all-directions
```

本轮复用了 `D:/Anaconda3/envs/aglldiff/python.exe` 与仓库已有 `outputs/hphy_deps`，版本为 PyTorch 2.3.0、MMCV 1.4.0、Matplotlib 3.7.5、pytest 8.3.5。下载依赖时曾遇到连接/TLS 错误，随后使用既有依赖完成验证。本地现有环境保持原安装状态。

Windows 沙箱默认临时目录使 MMCV 的 `NamedTemporaryFile` 创建受阻，诊断栈定位后将当前验证进程的 TMP/TEMP 指向 `outputs/irrg_test_tmp`。正式配置保留原训练协议。

## H. 八组训练命令和输出目录

完整批量执行入口如下。

```bash
bash scripts/run_doris_tableiv_26irrg_to_30rgb_hphy15k.sh
```

每条命令都指定独立绝对输出目录。

```bash
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi"

python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_phy_gate"

python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_struct_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_struct_gate"

python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate"

python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_ps_proto.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_ps_proto"

python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_struct_edge.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_struct_edge"

python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_pbcr.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_pbcr"

python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge"
```

## I. 4090 与真实 Checkpoint 待验收项

本地 CUDA 可用性为 False，真实数据和新训练 Checkpoint 均未提供。真实 MiT-B5 六通道前后向、实际验证入口、在线六样本采集负担、Checkpoint 写入、以及各分支真实 Checkpoint 回放待 4090 环境验收。

下列短 smoke 使用完整 HRDA/DACS 模型、随机初始化和受控 Teacher 分类器偏置，验证三个诊断迭代点。它会在独立输出目录生成合成验收数据，并可初始化真实数据探针。正式配置文件仍使用既定预训练初始化及 15000 Iter。

```bash
python tools/smoke_doris_hphy.py --direction goji26IRRG2goji30RGB --experiments 1 2 3 4 5 6 7 8 --output /root/shared-nvme/irrg26_rgb30_smoke_20261113 --check-fixed-probes
```

本次实施交付本地代码、测试、数据预检和 smoke 准备。正式 15K 训练由上述独立脚本启动。
