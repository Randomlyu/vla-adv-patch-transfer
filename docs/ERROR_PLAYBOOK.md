# 首日报错预案表（P0 基线）

> 依据：`PATCHES.md` 第 5 节「需上云验证」的 8 条，加 2 条路径/设备专项，共 10 条。
> 每条给：典型症状 / 一行根因 / 可直接粘贴的修法 / 兜底方案。
> 执行环境：AutoDL 单卡 4090，工作目录 `/root/autodl-tmp/vla-adv-patch-transfer`，conda env `vla-p0`。

## 索引

| # | 主题 | 触发阶段 | 严重度 |
|---|---|---|---|
| D-01 | flash-attn 2.5.5 与 torch/CUDA 不匹配 | `import torch` 后的首次 forward | 阻断 |
| D-02 | tensorflow-cpu 与 numpy 2.x 冲突 | `import tensorflow` | 阻断 |
| D-03 | 初始状态条数少于 trials | 进入 episode 循环 | 降级（有 clamp） |
| D-04 | EGL / OSMesa 离屏渲染不可用 | `get_libero_env()` | 阻断 |
| D-05 | 24 GB 显存不足 | 首次 `generate()` | 阻断 |
| D-06 | norm_stats 与套件不匹配 | 模型加载后断言 | 阻断 |
| D-07 | 贴片像素被裁切（全白/全黑） | 首条 attacked 的 mp4 | 数据失真 |
| D-08 | mp4 无法写出或无法回读 | 保存 rollout 视频 | 交付不完整 |
| D-09 | `Path(__file__).parents[3]` 未解析到仓库根 | 脚本启动 | 阻断 |
| D-10 | `cuda:0` 与 `CUDA_VISIBLE_DEVICES` 冲突 | 模型加载 | 阻断 |

---

## D-01 flash-attn 2.5.5 与 torch/CUDA 不匹配

**典型症状**

- `ImportError: .../flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c10...`
- `RuntimeError: FlashAttention only supports Ampere GPUs or newer`
- 进程在第一次 `generate()` 时直接 `Segmentation fault`，Python 层无 traceback

**一行根因**：`flash-attn` 二进制编译时的 torch/CUDA ABI 与当前解释器不一致（常见于先装了别的 torch 版本，再装 flash-attn）。

**修法**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip uninstall -y flash-attn
pip install --no-build-isolation --no-cache-dir ninja packaging
pip install --no-build-isolation --no-cache-dir "flash-attn==2.5.5"
```

**兜底**

```bash
# 源码编译，降低并行度避免 OOM
MAX_JOBS=4 pip install --no-build-isolation --no-cache-dir "flash-attn==2.5.5"
```

仍失败时，临时把 `experiments/robot/openvla_utils.py:45` 的 `attn_implementation="flash_attention_2"` 改为 `"sdpa"`。这是一次**行为变更**（注意力实现不同，数值可能有微小差异），必须在 `config.json` 之外的实验记录里写明；恢复原值后才算回到基线。

---

## D-02 tensorflow-cpu 与 numpy 2.x 冲突

**典型症状**

- `AttributeError: module 'numpy' has no attribute 'object'`（或 `'int'`、`'float'`）
- `ImportError: dlopen(... tensorflow ...): undefined symbol`
- `pip install` 时提示 `tensorflow 2.15.0 requires numpy<2, but you have numpy 2.x`

**一行根因**：TF 2.15 只支持 numpy 1.x，而 `pip install -e .` 或其它包把 numpy 顶到了 2.x。

**修法**

```bash
pip install "numpy==1.26.4" "tensorflow-cpu==2.15.0"
python -c "import numpy, tensorflow as tf; print(numpy.__version__, tf.__version__)"
```

**兜底**

```bash
pip install "numpy<2"
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.zeros([2,2])))"
```

TF 在 P0 里只用于 `tf.image.resize` 与 `center_crop`（`libero_utils.py:42-45`、`openvla_utils.py:139-151`），走 CPU 即可，不需要 GPU 版。

---

## D-03 初始状态条数少于 trials

**典型症状**

- `[warn] task 0: num_trials_per_task=50 exceeds available init states (M); clamped to M`
- 旧脚本上的 `IndexError: index out of range in self`

**一行根因**：`*.pruned_init` 里实际条数少于请求的 trials 数（`get_task_init_states()` 只做 `torch.load`，没有数量断言）。

**修法（30 秒探针，先拿到真实条数）**

```bash
python - <<'PY'
from libero.libero import benchmark
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    bench = benchmark.get_benchmark_dict()[suite]()
    print(suite, "tasks=", bench.n_tasks, "init_states=", len(bench.get_task_init_states(0)))
PY
```

**兜底**：首日固定 `--max_tasks 1 --num_trials_per_task 1`，不受任何条数限制；报 `[warn] clamped` 属于预期行为，不算失败。

---

## D-04 EGL / OSMesa 离屏渲染不可用

**典型症状**

- `GLException: Failed to open display`
- `MUJOCO_GL=egl` 下报 `EGL_NOT_INITIALIZED` / `Renderer: egl not available`
- `ImportError: libGL.so.1: cannot open shared object file`

**一行根因**：容器内没有可用的 GPU/软件渲染后端，或缺少系统库。

**修法**

```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0 libosmesa6-dev
export MUJOCO_GL=osmesa
python -c "from libero.libero.envs import OffScreenRenderEnv; print('env import ok')"
```

**兜底**

```bash
apt-get install -y xvfb
xvfb-run -a python experiments/robot/libero/run_libero_eval_args_geo_batch.py --help
```

首日验证顺序：`egl` → `osmesa` → `xvfb`，命中即停，把命中的那个写进后续所有命令的环境变量。

---

## D-05 24 GB 显存不足

**典型症状**

- `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.2 GiB`
- 模型加载到一半卡住，`nvidia-smi` 显示显存占满但 GPU 利用率 0

**一行根因**：7B bf16 权重约 15.1 GB，加上视觉塔、KV cache 与 MuJoCo 上下文，在 24 GB 上余量约 8 GB。

**修法**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

确认没有残留进程（上一次被 kill 的 python 可能仍占显存）：

```bash
ps -ef | grep run_libero_eval | grep -v grep
pkill -f run_libero_eval_args_geo_batch.py
sleep 5 && nvidia-smi
```

**兜底**：加 `--load_in_4bit`（bitsandbytes 路径，`openvla_utils.py:47-48`），这是**行为变更**，必须与 bf16 基线分开记录；或换 32 GB / 40 GB 卡。

---

## D-06 norm_stats 与套件不匹配

**典型症状**

```
AssertionError: Action un-norm key libero_object not found in VLA `norm_stats`!
available keys: ['libero_spatial']
```

**一行根因**：每个 LIBERO 微调 checkpoint 的 `dataset_statistics.json` 只带自己那一个 key，套件与 checkpoint 必须成对。

**修法**：按套件换 checkpoint。

```bash
python experiments/robot/libero/run_libero_eval_args_geo_batch.py \
  --pretrained_checkpoint /root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --max_tasks 1 --num_trials_per_task 1 --clean --run_id p0-clean-object-t0
```

**兜底**：首日只跑 `libero_spatial` + `openvla-7b-finetuned-libero-spatial` 这一对，跨套件属第二阶段（需要 `--unnorm_key` 与 suite 解耦，本单不做）。

---

## D-07 贴片像素被裁切（全白/全黑）

**典型症状**

- mp4 里贴片区域是一块纯白或纯黑方块，看不出图案
- clean 与 attacked 的 `metrics.jsonl` 除 `patch_path` 外完全一致，`success`/`steps` 都一样

**一行根因**：`simulation_random_patch` 用 `ToPILImage` 把 float 张量转 uint8（`appply_random_transform.py:60`），而该 API 假定 float 取值在 [0,1]；越界会被裁到 0 或 255。

**修法（先取事实）**

```bash
python -c "import torch; p=torch.load('adversarial_patches/simulation/untargeted/UPA-dof1~3-b9972dd8-8c22-4a2b-923a-92ee14d96e74/patch.pt'); print(p.shape, p.dtype, float(p.min()), float(p.max()))"
```

期望：`torch.Size([3, 50, 50]) torch.float32 0.0 1.0` 之间的值。

**兜底**：若 min/max 不在 [0,1]，先落一份归一化副本再跑（不改仓库文件，产物放 `runs/` 外或 `logs/`），并在实验记录里注明用的是副本。

```bash
python - <<'PY'
import torch
src = "adversarial_patches/simulation/untargeted/UPA-dof1~3-b9972dd8-8c22-4a2b-923a-92ee14d96e74/patch.pt"
p = torch.load(src).float()
print("before", float(p.min()), float(p.max()))
p = (p - p.min()) / (p.max() - p.min() + 1e-8)
torch.save(p, "/root/autodl-tmp/patch_norm.pt")
print("saved /root/autodl-tmp/patch_norm.pt")
PY
```

---

## D-08 mp4 无法写出或无法回读

**典型症状**

- `OSError: No ffmpeg exe could be found`
- `imageio` 写出的文件 0 字节；或 `[warn] no frames captured; replay video skipped`

**一行根因**：缺 ffmpeg 后端，或该条 episode 在取到第一帧之前就抛异常（`replay_images` 为空）。

**修法**

```bash
pip install "imageio[ffmpeg]"
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```

**兜底**：`[warn] no frames captured` 属于异常路径的正常输出，去看同目录 `metrics.jsonl` 该行的 `exception` 字段；不要为了拿到 mp4 反复重跑。

---

## D-09 `Path(__file__).parents[3]` 未解析到仓库根

**典型症状**

```
ModuleNotFoundError: No module named 'appply_random_transform'
```
或 `ModuleNotFoundError: No module named 'experiments'`

**一行根因**：脚本被复制/软链到别处，或仓库目录层级被改动，导致 `parents[3]` 不是仓库根。

**修法（30 秒探针，先验证再跑）**

```bash
python - <<'PY'
from pathlib import Path
f = Path("experiments/robot/libero/run_libero_eval_args_geo_batch.py").resolve()
print("file       :", f)
print("parents[3] :", f.parents[3])
print("has prismatic :", (f.parents[3] / "prismatic").is_dir())
print("has VLAAttacker:", (f.parents[3] / "VLAAttacker").is_dir())
print("has white_patch:", (f.parents[3] / "VLAAttacker" / "white_patch" / "appply_random_transform.py").is_file())
PY
```

在 `/root/autodl-tmp/vla-adv-patch-transfer/` 结构下，期望输出：

```
parents[3] : /root/autodl-tmp/vla-adv-patch-transfer
has prismatic : True
has VLAAttacker: True
has white_patch: True
```

**兜底**：若 `has white_patch` 为 `False`，说明仓库不完整（clone 中断或文件缺失），重新 `git clone`，不要在缺文件的状态下继续排错。

---

## D-10 `cuda:0` 与 `CUDA_VISIBLE_DEVICES` 冲突

**典型症状**

- `RuntimeError: Invalid device ordinal`
- 或指定了 `CUDA_VISIBLE_DEVICES=1` 却仍然只用第一张卡

**一行根因**：`CUDA_VISIBLE_DEVICES` 是**重映射**而不是"选择绝对编号"：设为 `1` 时，物理卡 1 在进程内变成 `cuda:0`。所以代码里固定 `cuda:0` 与任何 CVD 取值都不冲突。

**修法（一条命令确认）**

```bash
python -c "import os, torch; print('CVD=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('count=', torch.cuda.device_count()); print('name0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

单卡实例期望：`count= 1`、`name0= NVIDIA GeForce RTX 4090`。若 `count= 0`，是驱动/CVD 设成了不存在的编号，`unset CUDA_VISIBLE_DEVICES` 后重试。

**兜底**：不改代码里的 `cuda:0`，只改 `CUDA_VISIBLE_DEVICES`；不要引入第二套设备选择机制（`--cudaid` 已标注 deprecated 且不生效）。
