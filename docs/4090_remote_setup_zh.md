# PSSAL-DA：RTX 4090 远端试跑与上传目录

适用范围：当前仓库 `doris` 分支中的 Goji/NGMSeg 实验，先验证
26RGB→30RGB，再验证论文使用的完整模块和其他迁移方向。
这是候选环境和操作说明，尚未在你的远端 4090 上验收。

## 1. 判断依据

- 历史依赖：Python 3.8、torch 1.7.1+cu110、torchvision 0.8.2、
  mmcv-full 1.3.7。当前 `mmseg/__init__.py` 允许 MMCV 1.3.7～1.4.0。
- 不建议直接向平台预装的现代 PyTorch 环境安装旧 requirements。
  先在容器内建立独立 Python 3.8 环境，保留平台驱动。
- 第一候选：torch 1.13.1+cu117、torchvision 0.14.1+cu117、
  **mmcv 1.4.0（不带 full）**、timm 0.4.12。
  当前本仓库代码没有直接导入 `mmcv.ops`；先验证普通 MMCV 的
  runner/cnn/parallel 路线，可以避免编译 CUDA 扩展。
  此路线是否完整覆盖实际模型，仍以导入和训练结果为准。
- 不要同时安装 mmcv、mmcv-full；也不要安装 MMCV 2.x 或 pip 版
  MMSegmentation 来覆盖仓库自己的 `mmseg/`。
- CUDA 11.7 并不等于不能使用 4090：NVIDIA 说明兼容的 Ampere
  cubin 或 PTX 可以在 Ada 上运行。是否包含所需内核，由真实前后向验证。
  不要求 `torch.cuda.get_arch_list()` 必须出现 `sm_89`。
- 平台的“PyTorch25.03-Ubuntu22.04”是镜像名称，不能据此推断实际
  Python/Torch/CUDA。NVIDIA 官方 25.03 镜像是 Ubuntu 24.04、
  Python 3.12、CUDA 12.8.1、PyTorch 2.7.0a0；先执行下面的环境盘点。

## 2. 上传到哪里

本地已经创建 `D:\PSSAL-DA\shared-nvme` 空目录模板。
将其**内容**对应上传到远端 `/root/shared-nvme/`，不要再套一层
`shared-nvme/shared-nvme`。Git 不保存空目录，且该目录已被 `.gitignore`
忽略；仅同步 Git 代码不会同步这份目录模板。

```text
/root/shared-nvme/
├── pretrained/
│   ├── mit_b5.pth                 # MiT-B5 骨干的预训练权重
│   └── resnet101-5d3b4d8f.pth     # ResNet-101 实验配置指定的预训练权重
├── mydataset/
│   ├── 26/
│   │   ├── img_dir/{train,val,test}/    # IRRG 三通道 .tif
│   │   ├── rgb_dir/{train,val,test}/    # RGB 三通道 .tif
│   │   ├── ann_dir/{train,val,test}/    # 单波段类别 ID .tif
│   │   ├── ndvi_dir/{train,val,test}/   # 单波段 NDVI .tif
│   │   ├── ndre_dir/{train,val,test}/   # 单波段 NDRE .tif
│   │   └── cire_dir/{train,val,test}/   # 单波段 CIre .tif
│   └── 30/                            # 与 26 完全相同的六组子目录
└── work_dirs/                         # 自动生成日志和训练权重，不上传数据
```

上面的 `{train,val,test}` 表示三个独立文件夹，不是文件夹的字面名称。
也可以直接在远端 Bash 建目录：

```bash
mkdir -p /root/shared-nvme/pretrained /root/shared-nvme/work_dirs
for area in 26 30; do
  for modality in img_dir rgb_dir ann_dir ndvi_dir ndre_dir cire_dir; do
    mkdir -p /root/shared-nvme/mydataset/"$area"/"$modality"/{train,val,test}
  done
done
```

上传的是**原论文已经切好的 256×256 patch 数据**，保留原 train/val/test
划分、文件名、像素值与指数计算方式。同一个样本在六种目录下应同名，
例如 `26/rgb_dir/train/1_0_0_256_256.tif` 对应
`26/ann_dir/train/1_0_0_256_256.tif` 和三个指数文件。文件后缀为小写 `.tif`。
指数应是数值 TIFF，不是伪彩色截图；不要把指数先缩放到 0～255。

只试 26RGB→30RGB 时使用 26 的 RGB/标签/指数训练集，以及 30 的
RGB/标签/指数训练、验证、测试集；IRRG 是另外两组消融方向所需。
当前 target pipeline 仍会读取 `ann_dir/train`，因此也要同步该目录；
这只是当前加载器的文件要求，并不代表已审计 UDA 中目标标签的使用。

重要的标签核对：当前 `GojiDataset` 与 `LoadAnnotations` 使用
`reduce_zero_label=True`，要求磁盘有效类别是 **1～5**，0/255 被忽略，
加载后变成 0～4。`tools/convert_datasets/goji.py` 的 `remap_label()`
却会预先把 1～5 减成 0～4。不要为了环境迁移重新执行该转换脚本；
若已有标签确实是 0～4，先核对旧实验加载方式，不能直接再次减一，
也不要凭单张 patch 缺少类别 5 就认定标签已转换。

MiT-B5 路线上传 `mit_b5.pth`；ResNet-101 路线还需要上传
`resnet101-5d3b4d8f.pth`，两者均放在 `/root/shared-nvme/pretrained/`。
仓库的 MiT-B5 已有从 3 通道预训练卷积扩展到 6 通道输入的实现。
当前 Goji 从头训练配置
不需要 GTA→Cityscapes 的 HRDA 训练权重，也不需要 Cityscapes/GTA/
SYNTHIA 数据。旧实验 checkpoint 仅在恢复/评估旧实验时另行上传，
不要替代对应骨干的预训练权重。

ResNet-101 的 `resnet_101`、`baseline_resnet`、`noDACS_resnet` 以及部分
对比方法配置中，仍指定旧位置
`/root/HRDA-master/pretrained/resnet101-5d3b4d8f.pth`。
在运行相应配置时，使用以下命令行选项覆盖为上传位置：

```bash
--options model.backbone.pretrained=/root/shared-nvme/pretrained/resnet101-5d3b4d8f.pth
```

这是传给 `python tools/train.py <对应ResNet配置>` 的选项片段，不能单独执行。
后文 `check_4090.py` 默认预检 MiT-B5 基线，不能替代 ResNet-101 的
实际权重加载与训练验收。

`train.txt`、`val.txt`、`test.txt`、`patch_metadata.csv` 如原数据已有，
可以随各区根目录一起上传；当前三组消融配置不设置 `split`，按目录扫描，
不强制依赖这些文件。保持论文原有划分，不按现有转换脚本重新分配。

## 3. 先盘点远端环境

以下全部为**远端 Linux Bash** 命令，不是在本地 PowerShell 中执行。
仓库路径以下以 `/root/PSSAL-DA` 为例，请换成真实位置。
先把新增的 `requirements-4090.txt` 和 `tools/check_4090.py` 同步到远端仓库。

```bash
cd /root/PSSAL-DA
nvidia-smi
cat /etc/os-release
command -v conda || true
python - <<'PY'
import sys
print('Python:', sys.version)
print('Executable:', sys.executable)
try:
    import torch
    print('Torch:', torch.__version__)
    print('Torch CUDA runtime:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('Capability:', torch.cuda.get_device_capability(0))
except ImportError as exc:
    print(exc)
PY
```

`nvidia-smi` 的 CUDA Version 表示驱动支持的 CUDA 上限，不要求它与
`torch.version.cuda` 完全相同。容器内不需要重装宿主机 NVIDIA 驱动，
这条普通 MMCV 路线也不需要用系统 `nvcc` 编译扩展。

## 4. 建立隔离环境

若已有 conda，直接执行下面的创建命令。若没有 conda，可先安装独立
Miniforge 到新路径（要求 Linux x86_64；已存在该路径时不要重复安装）：

```bash
curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/Miniforge3-pssal.sh
bash /tmp/Miniforge3-pssal.sh -b -p /root/miniforge3-pssal
source /root/miniforge3-pssal/etc/profile.d/conda.sh
```

```bash
conda create -p /root/shared-nvme/envs/pssal4090 python=3.8 pip=23.3.2 -y
conda activate /root/shared-nvme/envs/pssal4090
unset PYTHONPATH PYTHONHOME PIP_CONSTRAINT PIP_BUILD_CONSTRAINT
export PYTHONNOUSERSITE=1
cd /root/PSSAL-DA
python -m pip install setuptools==59.5.0 wheel==0.41.3
python -m pip install -r requirements-4090.txt
python -m pip check
```

这是相对历史环境的候选变更，旧 `requirements.txt` 已保留：Torch/
torchvision 升至 1.13.1/0.14.1，MMCV 使用允许范围内的普通 1.4.0，
timm 改为 0.4.12，避免早期 timm 对已移除 `torch._six.container_abcs`
的依赖；NumPy 1.22.4 与旧 SciPy/Python 3.8 配合；Pillow 9.5.0
避开 torchvision 对 Pillow 8.3.* 的排除；typing-extensions 为 4.5.0。
其余直接依赖尽量沿用旧版本。该文件不是已验收的完整依赖锁。

若 OpenCV 导入报 `libGL.so.1` / `libgthread-2.0.so.0` 缺失，在远端补系统库：

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0
```

安装失败时先保留完整错误，不要同时升级 Torch、MMCV、NumPy。
如果仍出现 25.03 镜像的版本约束冲突，检查 `python -m pip config debug`
及报错提到的 constraints 路径，确认隔离环境没有继承平台约束。

## 5. 分层预检

```bash
python tools/check_4090.py --env-only
python tools/check_4090.py --data-only
python tools/check_4090.py
```

- 第一条：真实 CUDA 运算、MMCV/DataContainer 搬运、卷积/矩阵乘法/
  插值/交叉熵反向传播和 AdamW，以及本仓库模型模块导入。
- 第二条：权重文件非空、两区全部六种目录的 TIFF 文件名匹配，
  每个 split 抽查三组 patch 的大小、通道、有限值与原始标签取值。
- 第三条：上述内容加实际配置解析、训练/验证 pipeline 取样。
  文件名是全量比对，TIFF 内容仅抽查；脚本不加载完整 HRDA 权重，
  不证明模型训练、所有 checkpoint 内容或全部数据质量已验收。

普通 MMCV 可能提示没有 `MultiScaleDeformableAttention`。该可选模块的
导入警告本身不等于本模型无法训练；若实际 traceback 调用了 `mmcv.ops`
或 `mmcv._ext`，应停在该错误定位依赖，再决定是否编译 mmcv-full。
不要用忽略异常或放宽版本断言掩盖必需算子缺失。

## 6. 先跑 20 iteration 基线

当前训练入口使用 **`--options`**，不是新版 MMSeg 的 `--cfg-options`。
三组 common 配置的预训练路径仍写死在
`/root/HRDA-master/pretrained/mit_b5.pth`；下面用命令行覆盖，原配置文件不改。
所有输出使用绝对路径，仓库不需要放在 `/root/HRDA-master`，也无需软链接。
公开版本 README 曾引用 `train_pssal_da.sh`，本次检查时该文件不存在，
以下直接使用已核对的 `tools/train.py` 入口。

```bash
cd /root/PSSAL-DA
RUN=/root/shared-nvme/work_dirs/compat4090_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u tools/train.py \
  configs/hrda/goji26RGB2goji30RGB_hrda_3vi.py \
  --work-dir "$RUN" --no-validate --seed 0 \
  --options \
  model.backbone.pretrained=/root/shared-nvme/pretrained/mit_b5.pth \
  runner.max_iters=20 data.workers_per_gpu=0 \
  log_config.interval=1 checkpoint_config.interval=20 \
  2>&1 | tee "$RUN/console.log"
```

检查真实日志中预训练权重成功加载、3→6 通道适配、iteration 到 20，
loss 有限、无 CUDA kernel/cuDNN 错误，并实际生成 `iter_20.pth`。
保持原 batch=2、crop、学习率与精度配置；workers=0 只用于简化首轮排查。
本命令不做验证，loss 大小也不能用于判断论文效果。

若发生 OOM，先看 `nvidia-smi` 是否还有别的进程。batch=1 只能作为
单独的定位实验，不能直接当作原论文设置复现成功。

## 7. 再验证完整模块、验证集、保存与恢复

基线通过后，用当前含完整模块的配置跑 100 iteration，并在末尾执行
一次**完整验证集**评估；验证耗时取决于验证集大小。

```bash
RUN_FULL=/root/shared-nvme/work_dirs/compat4090_full_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_FULL"
CUDA_VISIBLE_DEVICES=0 python -u tools/train.py \
  configs/hrda/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py \
  --work-dir "$RUN_FULL" --seed 0 \
  --options \
  model.backbone.pretrained=/root/shared-nvme/pretrained/mit_b5.pth \
  runner.max_iters=100 data.workers_per_gpu=0 \
  log_config.interval=10 checkpoint_config.interval=100 evaluation.interval=100 \
  2>&1 | tee "$RUN_FULL/console.log"
```

随后从该试验的 checkpoint 恢复到 101，核对恢复日志的 iteration/
optimizer 状态与输出文件：

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/train.py \
  configs/hrda/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py \
  --work-dir "$RUN_FULL" --resume-from "$RUN_FULL/iter_100.pth" --seed 0 \
  --options \
  model.backbone.pretrained=/root/shared-nvme/pretrained/mit_b5.pth \
  runner.max_iters=101 data.workers_per_gpu=0 \
  log_config.interval=1 checkpoint_config.interval=101 evaluation.interval=100 \
  2>&1 | tee "$RUN_FULL/resume_console.log"
```

100/101 iteration 只证明前期训练、评估和恢复的基本链路。
当前若干模块在 3000/5000/6000 iteration 后才开始更新或起作用，
并有延续到 10000 iteration 的 ramp-up。短跑没有覆盖这些后期逻辑。
要确认论文完整方法能运行，应在新目录按正式 20000 iteration 配置跑，
至少观察越过这些阈值后的真实模块 loss、指标和 checkpoint：

```bash
RUN_PAPER=/root/shared-nvme/work_dirs/paper4090_full_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_PAPER"
python -m pip freeze > "$RUN_PAPER/environment.txt"
git rev-parse HEAD > "$RUN_PAPER/git_commit.txt"
git diff > "$RUN_PAPER/source_changes.patch"
sha256sum /root/shared-nvme/pretrained/mit_b5.pth > "$RUN_PAPER/pretrained.sha256"
nvidia-smi > "$RUN_PAPER/gpu.txt"
CUDA_VISIBLE_DEVICES=0 python -u tools/train.py \
  configs/hrda/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py \
  --work-dir "$RUN_PAPER" --seed 0 \
  --options model.backbone.pretrained=/root/shared-nvme/pretrained/mit_b5.pth \
  2>&1 | tee "$RUN_PAPER/console.log"
```

正式运行会恢复配置默认的 workers=4 和原 iteration/评估/保存间隔。
不要从 100 iteration 诊断试验继续当作论文正式结果，因为它改变过
`max_iters`，也就改变了学习率日程。其他迁移方向也应各自先短跑，
不能由 26RGB→30RGB 成功推断它们全部通过。

更换 GPU/依赖后不承诺与 3090 历史结果逐位一致。补实验应固定数据
划分、seed、batch、学习率、训练步数和指标口径，并记录新环境；
建议至少选一个论文已有基线在新环境重跑，评估环境差异的影响。

## 8. 出错时提供什么

保留：第 3 节环境盘点输出、失败命令、完整 traceback（包含首次异常）、
对应 `console.log`、`pip check` 结果。如果只是预检通过，表述为
“预检通过”；只有实际模型训练/评估/恢复通过后，才确认对应层次适配。

## 官方参考

- [HRDA 上游环境与权重说明](https://github.com/lhoyer/HRDA)
- [NVIDIA Ada 兼容性说明](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html)
- [PyTorch 1.13.1 官方安装命令](https://pytorch.org/get-started/previous-versions/#v1131)
- [NVIDIA PyTorch 25.03 镜像说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html)
- [MMCV 1.4.0 可选算子的导入方式](https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/transformer.py)
