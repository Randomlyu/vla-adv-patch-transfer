# 首日报错预案表（P0 基线）

> 依据：`PATCHES.md` 第 5 节「需上云验证」的 8 条，加 2 条路径/设备专项，共 21 条（D-01…D-10 为原预案，D-11…D-21 为增量）。
> 每条给：典型症状 / 一行根因 / 可直接粘贴的修法 / 兜底方案。
> 执行环境：AutoDL 单卡 4090D 24G，工作目录 `/root/autodl-tmp/vla-adv-patch-transfer`，conda env `vla-p0`。

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
| D-11 | flash-attn `build_wheel` 空转 41 分钟 | `pip install flash-attn` | 阻断 |
| D-12 | `import wandb` 报 `Imports` ImportError | 脚本启动第一行 | 阻断 |
| D-13 | `robosuite...single_arm_env` 缺失 | `import libero` 后进 env | 阻断 |
| D-14 | `No module named 'dlimp'` | import prismatic RLDS | 阻断 |
| D-15 | `No module named 'tensorflow_datasets'` | `import dlimp` | 阻断 |
| D-16 | `cannot import name 'runtime_version'` | `import tensorflow_datasets` | 阻断 |
| D-17 | `No module named 'tensorflow_graphics'`（及同批缺包） | 进 RLDS 工具链 | 阻断 |
| D-18 | `Connection to huggingface.co timed out` / `LocalEntryNotFoundError` | `from_pretrained` 取 config | 阻断 |
| D-19 | `binding_utils.py:521` AssertionError（mujoco enum） | `get_libero_env()` | 阻断 |
| D-20 | 探针假阴性与已推翻的判断 | 自检 / 排错 | 误判源 |
| D-21 | 可忽略的警告与报错清单 | 全流程 | 噪音 |

> 2026-11-27 增量：新增 D-11…D-21 共 11 条（正文见文末「2026-11-27 增量」段）。原 D-01…D-10 的编号与内容未改动。

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

单卡实例期望：`count= 1`、`name0= NVIDIA GeForce RTX 4090 D`。若 `count= 0`，是驱动/CVD 设成了不存在的编号，`unset CUDA_VISIBLE_DEVICES` 后重试。

**兜底**：不改代码里的 `cuda:0`，只改 `CUDA_VISIBLE_DEVICES`；不要引入第二套设备选择机制（`--cudaid` 已标注 deprecated 且不生效）。

---

## 2026-11-27 增量（D-11 … D-21）

> 以下 11 条全部来自当日有卡容器实测。每条七字段：触发现象 / 根因 / 就地修法 / 验证 / 耗时 / 学术加速 / 别做什么。

## D-11 flash-attn `build_wheel` 空转 41 分钟

**触发现象**

- `pip install "flash-attn==2.5.5" --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple` 停在 `Building wheel for flash-attn (pyproject.toml) ... \`，41 分钟无任何新输出。
- `ps` 显示 pip PID 1498 存活 41:13、`%CPU 0.0`；build backend 子进程 PID 1591 的 cmdline 为 `/root/miniconda3/envs/vla-p0/bin/python3.10 .../pip/_vendor/pyproject_hooks/_in_process/_in_process.py build_wheel /tmp/tmpujyhmf8k`。
- `ss -tnp` 过滤 1498 / 1591 → 无外连记录。
- `grep -E "ninja|nvcc|cc1plus|ptxas|cicc"` 无任何编译子进程。
- `python -c "import flash_attn"` → `ModuleNotFoundError`。
- AutoDL 面板 CPU 0%。

**根因**：`build_wheel` 子进程空转，未启动任何编译器，包未落地。
（「去 GitHub 找预编译 wheel」属于推测：`ss` 无外连记录，不作为定论。）

**就地修法（可复制）**

```bash
# 0) 判据：面板 CPU 0% + 无编译子进程 + 包未落地 = 真挂
ps -eo pid,etime,pcpu,cmd | grep -E "pip|ninja|nvcc|cicc|ptxas|cc1plus|g\+\+" | grep -v grep || echo "(no compile process)"

# 1) 从另一个已 conda activate vla-p0 的终端结束它（等价 Ctrl+C，不需要回到被占住的原终端）
kill -INT 1498

# 2) 后台强制本地编译
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
NJ=$(nproc); [ "$NJ" -gt 8 ] && NJ=8
nohup env FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=$NJ \
  pip install "flash-attn==2.5.5" --no-build-isolation --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  > /root/autodl-tmp/fa_build.log 2>&1 &
```

正常信号：`Building wheel ... started` + 连续 `still running...`；进程表出现 `ninja -v -j N` / `nvcc --generate-dependencies-with-compile` / `cicc --c++17`；CPU 真的上去；清华源 2.5 MB 以 22.6 MB/s、`0:00:00` 下载；一长串 `Requirement already satisfied` 证明 `--no-build-isolation` 生效、`torch 2.2.0+cu121` 未被重拉覆盖。

**验证**

```bash
python -c "import flash_attn; assert flash_attn.__version__=='2.5.5'; print('FA OK', flash_attn.__version__)"
```

期望：`FA OK 2.5.5`；日志尾部 `Successfully installed flash-attn-2.5.5` + `[1]+  Done`。

**耗时**：源码编译约 8–15 分钟（按 `nproc` 并发）。首次误用 `MAX_JOBS=2` 会慢到 20–30 分钟。

**学术加速**：全程关闭，零境外下载。

**别做什么**

- 不要以为「面板 CPU 0% 说明还在编译」——CPU 0% + 无编译进程 + 包未落地就是真挂。
- 不要让用户回原终端操作（原终端被前台进程占住），用另一个终端诊断。
- 不要用 `nvidia-smi` 利用率判断编译死活（编译是 CPU 活）。
- 不要在 `base` 环境里验证（Python 3.12、无 torch / flash-attn，必然 `ModuleNotFoundError`，会被误判成编译失败）。
- 不要 `MAX_JOBS=2`。

---

## D-12 `import wandb` 报 `Imports` ImportError

**触发现象**

- `run_libero_eval_args_geo_batch.py:38 import wandb` 启动即挂：`ImportError: cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2'`。
- 权重未加载、MuJoCo 未起、`runs/` 未建。

**根因**：该版本 wandb 的 sdk 与自带 proto 生成文件不自洽。

**就地修法（可复制）**

```bash
pip install --no-cache-dir "protobuf==4.25.3" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-cache-dir "wandb==0.16.6" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

已验证**无效**的做法：`wandb==0.31.2` 同版本 `pip install --force-reinstall --no-deps --no-cache-dir "wandb==0.31.2"` 后报错一字不差 → 换同版本 wheel 无用。

**验证**

```bash
python -c "import wandb; print('wandb OK', wandb.__version__)"
```

期望：`wandb OK 0.16.6`。

**耗时**：数十秒（清华源）。

**学术加速**：关闭。

**别做什么**

- 不要 `--force-reinstall` 同版本，已知无效。
- 不要用 `--no-deps` 装 wandb 0.16.6（要让它补齐自身小依赖；它不会碰 torch / TF）。
- 注意 protobuf 之后因 D-16 又降到 `3.20.3`，**最终态以 `DAY1_SOP.md` §0.6 为准**。
- P-08 当年只把 `--use_wandb` 默认改 False、entity 默认 None，**未去掉脚本顶层硬 import**；「把顶层 import 改成按需加载」是后续源码待办，本单不做。

---

## D-13 `robosuite...single_arm_env` 缺失

**触发现象**

```
ModuleNotFoundError: No module named 'robosuite.environments.manipulation.single_arm_env'
```

来自 `libero/libero/envs/bddl_base_domain.py:6 from robosuite.environments.manipulation.single_arm_env import SingleArmEnv`。

**根因**：装的 robosuite 版本目录结构不含 `single_arm_env.py`。实测装的是 `1.5.2`，`manipulation/` 下只有 `manipulation_env.py` / `two_arm_env.py` 等。本地 `/root/autodl-tmp/LIBERO/requirements.txt:10` 写的是 `robosuite==1.4.0`。

**就地修法（可复制）**

```bash
pip install --no-cache-dir --no-deps "robosuite==1.4.1" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**验证**

```bash
python -c "from robosuite.environments.manipulation.single_arm_env import SingleArmEnv; import robosuite; print('robosuite OK', robosuite.__version__, SingleArmEnv.__name__)"
```

期望：`robosuite OK 1.4.1 SingleArmEnv`。

**耗时**：193.5 MB / 18.1 MB/s，约十几秒。

**学术加速**：关闭（清华源）。

**别做什么**

- **笔误留痕**：派单写的是 1.4.1，LIBERO `requirements.txt` 写的是 1.4.0。1.4.1 为补丁版、目录结构兼容且实测通过，按不浪费时间原则**不回退**，文档注明差异即可。
- 不要装 1.5.x 再想办法打补丁，直接锁 1.4.1。

---

## D-14 `No module named 'dlimp'`

**触发现象**

```
prismatic/vla/datasets/rlds/dataset.py:13: ModuleNotFoundError: No module named 'dlimp'
```

**根因**：`dlimp` 不在 PyPI。OpenVLA 官方 `pyproject.toml` 声明为 `dlimp @ git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a`（其 `setup.py` 另声明 `tensorflow==2.15.0` + `tensorflow_datasets>=4.9.2`）；本仓库 `pyproject.toml` 漏掉这一项，与 `tensorflow-cpu` 被注释同类。

**就地修法（可复制，唯一需要开学术加速的一条）**

```bash
source /etc/network_turbo
pip install --no-cache-dir --no-deps "dlimp @ git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

**验证**

```bash
python -c "import dlimp; print('dlimp OK', dlimp.__file__)"
```

期望输出含 `Resolved ... to commit ad72ce3a9b414db2185bc0b38461d4101a65477a` / `Successfully built dlimp` / `Successfully installed dlimp-0.0.1`。

**耗时**：10–40 秒。

**学术加速**：**必须开**（GitHub）；装完立即 `unset`。

**别做什么**

- `--no-deps` 是硬约束：否则 pip 会去装 GPU 版 tensorflow，并可能把 numpy 拉到 2.x 打爆 gym 0.21。
- 不要省 `unset`，否则后续 PyPI / hf-mirror 会绕代理变慢。
- 兜底：若 `Connection timed out`，换 `moojink/dlimp_openvla` 镜像，仍走加速。

---

## D-15 `No module named 'tensorflow_datasets'`

**触发现象**

```
dlimp/dataset.py:7: ModuleNotFoundError: No module named 'tensorflow_datasets'
```

**根因**：`dlimp` 依赖 tfds，但上一步用 `--no-deps` 装，未带入。

**就地修法（可复制）**

```bash
pip install --no-cache-dir "numpy<2" "tensorflow-datasets==4.9.3" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

带入 `array-record 0.8.1` / `dm-tree 0.1.10` / `promise 2.3` / `tensorflow-metadata 1.21.0`。

**验证**

```bash
python -c "import tensorflow_datasets as tfds; print('tfds OK', tfds.__version__)"
```

期望：`tfds OK 4.9.3`。

**耗时**：数十秒。

**学术加速**：关闭（清华源）。

**别做什么**

- `dlimp 0.0.1 requires tensorflow==2.15.0, which is not installed` 是**分发名差异**（装的是 `tensorflow-cpu 2.15.0`，模块名 `tensorflow`，TF 已成功 import），无害，不要因此去装 GPU 版 TF。
- 不要顺手让 numpy 升到 2.x。

---

## D-16 `cannot import name 'runtime_version' from 'google.protobuf'`

**触发现象**

```
tensorflow_metadata/proto/v0/anomalies_pb2.py:9: from google.protobuf import runtime_version as _runtime_version
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

**根因**：pip 自动拉的 `tensorflow-metadata 1.21.0` 生成的 pb2 需要 `google.protobuf.runtime_version`，而 protobuf `4.25.3` 无此符号。

**就地修法（可复制）**

```bash
pip install --no-cache-dir --no-deps --force-reinstall \
  "protobuf==3.20.3" "tensorflow-metadata==1.14.0" "absl-py==1.4.0" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

三个版本取 TF 2.15 / tfds 4.9.3 / wandb 0.16.6 的 protobuf 约束交集。

**验证**

```bash
python -c "import google.protobuf as p, tensorflow_metadata as m; print('pb', p.__version__, 'tfmeta', m.__version__)"
```

期望：`pb 3.20.3 tfmeta 1.14.0`。

**耗时**：数十秒。

**学术加速**：关闭。

**别做什么**

- 不要为了让 `runtime_version` 存在而升 protobuf（会连带打断 wandb 0.16.6 与 TF 2.15）。
- `--no-deps` 必须带，否则会动 torch / numpy / TF。

---

## D-17 `No module named 'tensorflow_graphics'` 及「按官方清单一次补齐」

**触发现象**

```
prismatic/vla/datasets/rlds/oxe/utils/droid_utils.py:6: import tensorflow_graphics.geometry.transformation as tfg
ModuleNotFoundError: No module named 'tensorflow_graphics'
```

**根因**：仓库 `pyproject.toml` 把 TF 相关的几条注释掉了，运行时依赖实际缺一整批，不是单个包。

**就地修法（可复制）**：改为「按 OpenVLA 官方 `pyproject.toml` 运行时依赖建映射 → 用 `find_spec` 只装当前确实缺失的 → 全部 `--no-deps` → 干净子进程跑整条 import 链，从 stderr 抓 `No module named 'X'` 补装，最多 6 轮」。

| 模块名 | pip 包 |
|---|---|
| `tensorflow_graphics` | `tensorflow-graphics==2021.12.3` |
| `draccus` | `draccus==0.8.0` |
| `json_numpy` | `json-numpy` |
| `jsonlines` | `jsonlines` |
| `rich` | `rich` |
| `sentencepiece` | `sentencepiece==0.1.99` |
| `timm` | `timm==0.9.10` |
| `matplotlib` | `matplotlib` |
| `peft` | `peft==0.11.1` |
| `accelerate` | `accelerate>=0.25.0` |
| `tensorflow_datasets` | `tensorflow-datasets==4.9.3` |
| `dlimp` | `dlimp` |

```bash
python - <<'PY'
import importlib.util as u
want = {
    "tensorflow_graphics": "tensorflow-graphics==2021.12.3",
    "draccus": "draccus==0.8.0",
    "json_numpy": "json-numpy",
    "jsonlines": "jsonlines",
    "rich": "rich",
    "sentencepiece": "sentencepiece==0.1.99",
    "timm": "timm==0.9.10",
    "matplotlib": "matplotlib",
    "peft": "peft==0.11.1",
    "accelerate": "accelerate>=0.25.0",
    "tensorflow_datasets": "tensorflow-datasets==4.9.3",
}
missing = [pkg for mod, pkg in want.items() if u.find_spec(mod) is None]
print(" ".join(missing) if missing else "(nothing missing)")
PY
```

**验证（正确探针，不引用任何内部符号）**

```bash
python -c "import experiments.robot.openvla_utils as m; print('CHAIN OK', m.__file__)"
```

期望：`CHAIN OK /root/autodl-tmp/vla-adv-patch-transfer/experiments/robot/openvla_utils.py`。

**耗时**：每轮数十秒，最多 6 轮。

**学术加速**：关闭（清华源）。

**别做什么**

- **探针假阴性（必记）**：批量脚本写死 `from experiments.robot.openvla_utils import get_model`，而该模块实际不导出这个名字 → `ImportError: cannot import name 'get_model' from '...openvla_utils.py'`；循环只认 `No module named`，抓不到就判 `UNMAPPED` 退出。该报错恰恰说明模块已完整加载、整条 prismatic 依赖链已通。探针**不要引用任何内部符号**。
- 不要「报一个补一个」地逐个试，一次按清单补齐更快。

---

## D-18 `Connection to huggingface.co timed out` / `LocalEntryNotFoundError`

**触发现象**

```
AutoModelForVision2Seq.from_pretrained → AutoConfig → get_class_from_dynamic_module → cached_file
https://huggingface.co/openvla/openvla-7b/resolve/main/configuration_prismatic.py
Connection to huggingface.co timed out (connect timeout=10)
LocalEntryNotFoundError → OSError
```

此时模型尚未加载。

**根因**：finetuned ckpt 目录只有 13 个文件，**不带** `configuration_prismatic.py` / `modeling_prismatic.py` / `processing_prismatic.py`；这 3 个文件在仓库 `prismatic/extern/hf/` 里，于是 transformers 走「下载 dynamic module」的路径。

**就地修法（可复制，0 秒、纯本地、不联网）**

```bash
CKPT=/root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial
for f in configuration_prismatic.py modeling_prismatic.py processing_prismatic.py; do
  [ -f "$CKPT/$f" ] || cp "prismatic/extern/hf/$f" "$CKPT/$f"
done

python - <<'PY'
import json, os
CKPT = "/root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial"
fix = {
  "config.json": {
    "AutoConfig": "configuration_prismatic.OpenVLAConfig",
    "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
  },
  "preprocessor_config.json": {
    "AutoImageProcessor": "processing_prismatic.PrismaticImageProcessor",
    "AutoProcessor": "processing_prismatic.PrismaticProcessor",
  },
}
for fn, kv in fix.items():
    p = os.path.join(CKPT, fn)
    d = json.load(open(p))
    d.setdefault("auto_map", {}).update(kv)
    json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
    print(fn, "->", d["auto_map"])
PY
```

即把 `auto_map` 里 `repo_id--module.Class` 改成纯 `module.Class`。然后是所有 rollout 命令加：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/root/autodl-tmp/cache/
```

**验证**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('/root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial', trust_remote_code=True); print('LOCAL CONFIG OK', type(c).__name__)"
```

期望：`LOCAL CONFIG OK OpenVLAConfig`。

**耗时**：0 秒级。

**学术加速**：不适用。本地 `cp` 比任何联网方案都快；`HF_HUB_OFFLINE=1` 直接禁止 HTTP 调用，连 10 秒超时都不再等。

**别做什么**

- 这是**云端权重目录本地修改**，不在仓库、不进 git；**换 ckpt 需重做**。不要写成源码补丁，也不要列进 `PATCHES.md` 的 11 项。
- 不要为了绕超时去开 `network_turbo` 重下 15 GB。

---

## D-19 `binding_utils.py:521` AssertionError（mujoco enum 回归）

**触发现象**

- 模型已完整加载（`Loaded model: ...OpenVLAForActionPrediction`）、`metrics.jsonl` 已建、`planned episodes=1` 之后，进 `get_libero_env` 崩在 `robosuite/utils/binding_utils.py:521`。
- 栈尾：`assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)`。
- 当时实测 `mujoco 3.17.0`。

**根因**：MuJoCo ≥3.12 的 pybind11 3.1.0 改了 enum 比较方向，`mjtJoint.__eq__(numpy.int32)` 返回 False，机械臂所有铰链 / 滑动关节全部 assert 失败。3.10/3.11 另有 `MjData.qM` 缺失问题，3.9.0 可用。

**就地修法（可复制，两步）**

```bash
# 第一步：降级
pip install --no-cache-dir --no-deps --force-reinstall "mujoco==3.2.3" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 第二步：版本无关的 int() 补丁（robosuite 官方 issue 建议）
F="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/robosuite/utils/binding_utils.py"
sed -i 's/joint_type = self\.jnt_type\[joint_id\]/joint_type = int(self.jnt_type[joint_id])/g' "$F"
grep -c 'joint_type = int(self.jnt_type\[joint_id\])' "$F"
```

`int()` 把 numpy 标量转成 Python int，`mjtJoint == int` 在所有 MuJoCo 版本都成立。

**验证**

```bash
python -c "import mujoco; print('mujoco', mujoco.__version__)"
F="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/robosuite/utils/binding_utils.py"
grep -c 'joint_type = int(self.jnt_type\[joint_id\])' "$F"   # 期望 2
```

期望：`mujoco 3.2.3`，且 `grep -c` 返回 **2**（`get_joint_qpos_addr` / `get_joint_qvel_addr` 两处）。

**耗时**：降级下载 6.1 MB（40.3 MB/s）；`sed` 即时。

**学术加速**：关闭（清华源）。

**别做什么**

- **探针假阴性（必记）**：`python -c 'import numpy as np, mujoco; v=np.int32(0); ok=v in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE); print(...)'` 会打印 `ENUM_CMP_OK False` 并 assert 失败——因为 `np.int32(0)` 是 `mjJNT_FREE(0)`，而查的是 `mjJNT_HINGE(3)` / `mjJNT_SLIDE(2)`，拿 0 测 (3,2) 任何版本都 False。**这是探针写错，不是降级失败**。
- 这是**云端 site-packages 环境级修改**，不在仓库、不进 git，重装环境需重打；不得写成仓库源码改动、不得列入 `PATCHES.md` 的 11 项。
- 崩之前 `metrics.jsonl` 已建，重跑前先清残行：`mkdir -p runs/<run_id> && : > runs/<run_id>/metrics.jsonl`，保证验收时「恰 1 行」。

---

## D-20 探针假阴性与已推翻的判断

**触发现象**

同一现象被误判为「环境坏了」，实际是探针写法错、前提错或环境选错。

**根因（逐条客观记录，共七行；其中第 5、6 行是把「探针不得写死内部符号 / 不得用错误枚举值」这一条拆开写）**

| # | 已推翻的判断 | 实测事实 |
|---|---|---|
| 1 | 无卡模式「2 GB 内存编译必 OOM」 | 有卡容器实测 `Mem: 1.0Ti total / 900Gi available`，前提不适用；`MAX_JOBS` 应按 `min(nproc, 8)`，不是 2 |
| 2 | 前台进程占住终端时「回原终端贴命令」 | 原终端被占住，无法操作；应改用另一个已 `conda activate vla-p0` 的终端做诊断 / 验证。新开终端默认是 `base`（Python 3.12，无 torch / flash-attn），在其中 `import flash_attn` 必然 `ModuleNotFoundError`，会被误判成编译失败 |
| 3 | 「面板 CPU 0% 说明还在编译」 | 面板 CPU 0% + `ps` 无编译进程 + 包未落地 = 真挂；反之 CPU 有占用、`ninja/nvcc/cicc` 在列、`still running...` 在滚 = 正常 |
| 4 | 「GitHub 预编译 wheel 探测导致空转」 | 这是推测；`ss` 无外连记录，只记录可观测事实，不作定论 |
| 5 | 探针写死模块内部符号名 | `from experiments.robot.openvla_utils import get_model` 必然 `ImportError`（该模块不导出此名），正确写法是 `import ... as m; print(m.__file__)` |
| 6 | 用 `np.int32(0)` 测 `mjJNT_HINGE` / `mjJNT_SLIDE` | 0 是 `mjJNT_FREE`，测 (3,2) 任何版本都 False |
| 7 | `py_compile 99/99` / `compileall 全过` / `OK 2.2.0+cu121 2.15.0 [...]` = 环境全绿 | 这些的天花板是查语法与顶层 import，**查不出缺包、查不出运行时 enum 回归**；不得据此断言「环境全绿可直接跑」 |

**就地修法**：探针一律不引用内部符号、不用错误枚举值；判死活用进程表 + 日志 + 包是否落地三件套。

**验证**：见 D-17 / D-19 的正确探针命令。

**耗时**：不适用。

**学术加速**：不适用。

**别做什么**：不要把上表任一条当成「环境问题」去重装；先确认是探针错还是环境错。

---

## D-21 可忽略的警告与报错清单

**触发现象**：见下表。

| 输出 | 性质 | 处理 |
|---|---|---|
| `Do you want to specify a custom path for the dataset folder? (Y/N):` | 交互提示 | 回 `N` 回车。禁止 Ctrl+C、禁止重跑整条命令（重跑要重新加载 15.1 GB） |
| `[Warning]: datasets path .../libero/libero/../datasets does not exist!` | 路径提示 | 不影响评估（LIBERO 官方 issue #112 明确 "It seems not affect evaluation"） |
| `Exception ignored in: <function MjRenderContext.__del__ ...>` / `EGL_NOT_INITIALIZED (eglMakeCurrent)` | 退出析构阶段异常 | 发生在 `Saved rollout MP4` 与 `Success:` 打印**之后**。**不重跑、不换 osmesa、不换 xvfb**；`MUJOCO_GL=egl` 已实测能出视频 |
| `[robosuite WARNING] No private macro file found!` / `Could not import robosuite_models` / `Could not load the mink-based whole-body IK` | 可选组件缺失 | Panda 不依赖，忽略 |
| `Gym has been unmaintained since 2022 ... Please upgrade to Gymnasium` | 弃用警告 | **禁止升级**（v0.26+ 改 `reset()/step()` 返回值与 `env.seed()`） |
| `This TensorFlow binary is optimized to use ... AVX2 AVX512F ...` | CPU 指令集提示 | 不是报错 |
| `The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache.` | 一次性缓存迁移提示（写死在源码里） | 不是版本降级 |
| `<frozen importlib._bootstrap>:283: DeprecationWarning: the load_module() method is deprecated` | `trust_remote_code` 提示 | 无害 |
| `appply_random_transform.py:60: UserWarning: The given NumPy array is not writable` | 只读数组警告 | 转换路径不写回原数组，无害 |
| `WARNING: Running pip as the 'root' user...` / `ERROR: pip's dependency resolver does not currently take into account...` | pip 提示 | 本环境用 `--no-deps` 刻意隔离，后续 import 通过即忽略 |
| `libgomp: Invalid value for environment variable OMP_NUM_THREADS` | 镜像把变量设成非法值 | libgomp 自动回退、不影响 CUDA；`export OMP_NUM_THREADS=1` 覆盖 |

**根因**：多数是上游设计、可选组件或 pip 提示，不是本次评估链的故障。

**就地修法**：无需操作；按上表逐条对应即可。

**验证**：以 `metrics.jsonl` 是否 `invalid=false`、mp4 是否可回读为准。

**耗时**：不适用。

**学术加速**：不适用。

**别做什么**：**不重跑、不换后端、不升级 gym**。另：助手侧读网页索引产生的提示（形如资源解析失败一类）**不是云端终端输出**，不写入本文件、也不据此判断下载成败。

---

## 环境修复全景时序（2026-11-27）

| 顺序 | 现象 | 根因归类 | 修法所在条目 | 状态 |
|---|---|---|---|---|
| 1 | 六项前置自检（proxy / GPU / nvcc / torch / 权重计数 / libero 命名空间包） | 前置确认 | 本文件 D-20、`DAY1_SOP.md` §0.6 | 已实测通过 |
| 2 | flash-attn 停在 `Building wheel` 41 分钟 | 构建空转 | D-11 | 已实测通过 |
| 3 | `import wandb` → `Imports` ImportError | 版本不自洽 | D-12 | 已实测通过 |
| 4 | `robosuite...single_arm_env` 缺失 | 版本目录结构 | D-13 | 已实测通过 |
| 5 | `No module named 'dlimp'` | 仓库声明漏项 | D-14 | 已实测通过 |
| 6 | `No module named 'tensorflow_datasets'` | 依赖未带入 | D-15 | 已实测通过 |
| 7 | `cannot import name 'runtime_version'` | protobuf / tf-metadata 版本冲突 | D-16 | 已实测通过 |
| 8 | `No module named 'tensorflow_graphics'` 及同批缺包 | 仓库声明漏项（成批） | D-17 | 已实测通过 |
| 9 | `Connection to huggingface.co timed out` / `LocalEntryNotFoundError` | 权重目录缺 dynamic module | D-18 | 已实测通过 |
| 10 | `binding_utils.py:521` AssertionError | mujoco enum 比较回归 | D-19（降级 3.2.3） | 已实测通过 |
| 11 | 同一条 assert，改版本无关写法（不依赖具体 mujoco 版本） | 跨版本恒成立 | D-19（两处 `int()`，`grep -c` = 2） | 已实测通过 |
| 12 | clean 1 条 / attacked 1 条 | 链路验收 | `DAY1_SOP.md` §4.5 S12 / §5.2 | 已实测通过 |
