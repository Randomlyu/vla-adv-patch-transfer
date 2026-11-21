# PATCHES.md — P0 基线可运行最小修复包

> 目标：在 AutoDL 单卡 4090 上跑完「clean 1 episode + attacked 1 episode」，产出 rollout mp4 与
> `runs/<run_id>/metrics.jsonl`。
> 基线：`roboticAttack` commit `a0bef50` → 本 fork 基线 commit `19d2e5a`。
> 本文件只记录"为了能跑"所做的改动，不包含任何实验结论。

---

## 1. 前置确认结果（动手前实测）

| 项 | 实测结果 | 处理 |
|---|---|---|
| `git remote -v` | `origin https://github.com/Randomlyu/vla-adv-patch-transfer.git`（fetch/push 均指向自建仓库） | `.git` 已指向自建仓库，**未执行**"断开原 remote → 重新 init" |
| 分支 | 仅 `main`（本地与 `origin/main` 同步） | 直接在当前分支上改 |
| 基线 commit | `19d2e5a 初始化：基于 roboticAttack commit a0bef50，准备修复与迁移性分析` | 以此为父提交 |
| 工作区 | `git status --short` 为空 | 改动前的干净基线 |
| `adversarial_patches/` 是否在仓库内 | `git ls-files adversarial_patches \| wc -l` = **52**（26 组 `patch.pt` + `patch.png`） | 已在 git 内，云上 clone 即可用 |
| 是否被 `.gitignore` 排除 | `git check-ignore -v adversarial_patches/.../patch.pt` 退出码 **1**（未命中规则） | 未被排除 |
| 跟踪文件总量 / `.git` 体积 | 263 个文件 / 30 MB | `videos/` 的 99 个 mp4 已跟踪，保持原状 |
| `patch.pt` 格式（静态解析，未加载 torch） | torch zip 容器，成员 `patch/data.pkl`、`patch/data/0`；pickle 中 `torch FloatStorage`、7500 个元素、`BININT 3/50/50` | 形状 **(3, 50, 50) float32**，与 `patch_size="3,50,50"` 一致 |

结论：不需要重新初始化仓库；贴片已随仓库分发，attacked rollout 的输入齐备。

---

## 2. 必改项（按执行顺序）

### P-01 贴片模块缩进错误（必须改）

**文件**：`VLAAttacker/white_patch/appply_random_transform.py:43`

改前（3 空格，与类体 4 空格不一致，`py_compile` 直接 `IndentationError`）：

```python
   def simulation_random_patch(self, image, patch, geometry=False,colorjitter=False,angle=1,shx=0.1,shy=0.1,position=(0,0)):
```

改后（4 空格）：

```python
    def simulation_random_patch(self, image, patch, geometry=False,colorjitter=False,angle=1,shx=0.1,shy=0.1,position=(0,0)):
```

**原因**：该文件是评估链路唯一的贴片叠加实现，import 失败则整条链路不可运行。修复前这是全仓 98 个 py 中唯一的语法错误。

---

### P-02 视觉设备号硬编码 `cuda:1`（必须改，清单外新增）

**文件**：`experiments/robot/openvla_utils.py:21`

改前：

```python
DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
```

改后：

```python
# Device selection is driven by CUDA_VISIBLE_DEVICES (see README); inside the process the
# only visible GPU is always index 0. A hard-coded "cuda:1" raises "invalid device ordinal"
# on single-GPU hosts.
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
```

**原因**：AutoDL 单卡实例只暴露一张卡，进程内合法索引只有 0。该常量同时被 `get_vla()` 的 `vla.to(DEVICE)` 与 `get_vla_action()` 的 `processor(...).to(DEVICE)` 使用，`cuda:1` 会在加载权重阶段直接抛错。此问题不在原清单中，属阻塞 baseline 的隐性问题。

**设备选择结论**：只用 `CUDA_VISIBLE_DEVICES` 一个机制；`--cudaid` 保留参数位但显式标注为 deprecated 且不生效，避免出现两套设备选择逻辑。

---

### P-03 占位 `sys.path`（必须改）

**文件**：`experiments/robot/libero/run_libero_eval_args_geo_batch.py:32-34`（原行号；新文件 `:37-45`）

改前：

```python
import sys
sys.path.append("PATH TO/white_patch")
from appply_random_transform import RandomPatchTransform
```

改后：

```python
# Repo root = three levels up from <repo>/experiments/robot/libero/this_file.py
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "VLAAttacker" / "white_patch"))

from appply_random_transform import RandomPatchTransform  # noqa: E402
```

同时删除原 `:41` 的相对路径 `sys.path.append("../..")`（依赖当前工作目录，换目录即失效），并移除已无用的 `import draccus`（`GenerateConfig` 整段处于注释状态）。

**原因**：`"PATH TO/white_patch"` 是占位符，任何调用都在 import 阶段失败；`../..` 依赖 cwd，与 README 里"从仓库根执行"的约定冲突。

---

### P-04 `get_model` 多传 `DEVICE`（必须改）

**文件**：同文件 `:114`（新 `:175`）

改前：

```python
model = get_model(cfg,DEVICE=f"cuda:{cfg.cudaid}")
```

改后：

```python
model = get_model(cfg)
```

**原因**：`experiments/robot/robot_utils.py:40` 的签名是 `get_model(cfg, wrap_diffusion_policy_for_droid=False)`，没有 `DEVICE` 关键字，多余关键字会 `TypeError`。设备已在 P-02 统一。

---

### P-05 `get_action` 多传 `DEVICE`（必须改）

**文件**：同文件 `:227`（新 `:323`）

改前：

```python
action = get_action(
    cfg, model, observation, task_description,
    processor=processor,
    DEVICE=f"cuda:{cfg.cudaid}",
)
```

改后：

```python
action = get_action(
    cfg, model, observation, task_description,
    processor=processor,
)
```

**原因**：`robot_utils.py:63` 签名为 `get_action(cfg, model, obs, task_label, processor=None)`，无 `DEVICE` 参数。

---

### P-06 贴片路径改为显式参数（必须改；改后用参数绕开队列脚本）

**文件**：同文件 `:332`（新 `:524-530`）

改前：

```python
parser.add_argument("--patchroot", type=str, default="/spl_data/tw9146/openvla-main/run/white_patch_attack/a5083c2b-1186-4464-ab9f-1056211a2221/4000/patch.pt", help="")
```

改后：

```python
parser.add_argument("--patch_path", type=str, default=None,
                    help="Path to a pre-generated patch.pt. Omit (or pass 'none') for a clean run.")
parser.add_argument("--clean", action="store_true", help="Force a clean run; ignores --patch_path")
parser.add_argument("--patchroot", type=str, default=None, help="Deprecated alias of --patch_path")
```

解析与加载（新 `:79-107`, `:160`）：

```python
def _resolve_patch_path(raw):
    """Map the CLI value to a patch file path, or to None for a clean run."""
    if raw is None:
        return None
    value = str(raw).strip()
    if value == "" or value.lower() in {"none", "null", "clean", "off"}:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--patch_path does not exist: {path}")
    return path


def _load_patch(path):
    """Load a pre-generated patch tensor as (3, H, W) float32 on CPU."""
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("patch", "adv_patch", "tensor"):
            if key in obj:
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise TypeError(f"patch file does not contain a tensor: {path} (got {type(obj)})")
    if obj.dim() != 3 or obj.shape[0] != 3:
        raise ValueError(f"expect patch shape (3, H, W), got {tuple(obj.shape)} from {path}")
    if obj.dtype != torch.float32:
        obj = obj.float()
    return obj.contiguous()

# in eval_libero
patch_path = None if cfg.clean else _resolve_patch_path(cfg.patch_path or cfg.patchroot)
mode = "clean" if patch_path is None else "attacked"
patch = _load_patch(patch_path) if patch_path is not None else None
```

**原因**：原默认值是仓库外绝对路径，必然 `FileNotFoundError`。
**绕开方式**：`evaluation_tool/eval_queue_single_four_spec.py:32` 用 `os.listdir(exp_path)[0]` 取唯一子目录，26 组贴片只会跑到 1 组。P0 不再调用该队列脚本，直接用 `--patch_path` 指向具体 `patch.pt`。
**clean 运行**：`--clean` 或 `--patch_path none`。原链路没有"无贴片"分支，clean baseline 无法执行，此项为必需新增。

---

### P-07 异常被静默吞掉（必须改，优先级最高）

**文件**：同文件 `:246-249`（新 `:345-355`）

改前：

```python
except Exception as e:
    print(f"Caught exception: {e}")
    log_file.write(f"Caught exception: {e}\n")
    break
```

改后（对外层 `try` 捕获整条 episode，不再逐 step 吞掉）：

```python
except Exception as err:  # noqa: BLE001 - every failure must be recorded, not swallowed
    invalid = True
    exception = f"{type(err).__name__}: {err}"
    traceback_text = traceback.format_exc()
    print(traceback_text, file=sys.stderr)
    log_file.write(traceback_text)
    log_file.flush()
```

收尾（新 `:468-476`）：

```python
metrics.close(summary)

if invalid_episodes:
    print(
        f"[fatal] {invalid_episodes}/{total_episodes} episode(s) raised exceptions; "
        f"rows with invalid=true in {run_dir / 'metrics.jsonl'}",
        file=sys.stderr,
    )
    sys.exit(2)
```

**原因**：原实现只打印 `str(e)` 且随即 `break`，异常被计为一次"失败 episode"，FR 被污染且看不到 traceback。改后：完整 traceback 进 stderr 与 `.log`，该条 `metrics.jsonl` 写 `invalid=true`、`success=false`、`exception="类型: 信息"`，进程最终以退出码 2 结束。`success` 只在 `done and not invalid` 时计为真。

---

### P-08 W&B 默认开启 + `type=bool`（必须改，只保留一套机制）

**文件**：同文件 `:139-144` 与 `:328`（新 `:223` 与 `:544`）

改前：

```python
if cfg.use_wandb:
    wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
...
parser.add_argument("--use_wandb", type=bool, default=True, ...)
parser.add_argument("--wandb_entity", type=str, default="taowen_wang-rit", ...)
```

改后：

```python
if cfg.use_wandb:
    wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
...
parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases (off by default)")
parser.add_argument("--wandb_entity", type=str, default=None)
```

**结论（单一机制）**：`--use_wandb` 改为 `store_true` 且默认 **False**；不引入 `WANDB_MODE=offline`，不保留两套开关。需要上报时显式传 `--use_wandb`，并要求外部已配置 API key。
**原因**：`type=bool` 对字符串恒为真，命令行传 `"False"` 仍会初始化 W&B；原 entity 是作者个人账号，未登录环境下 `wandb.init` 会阻塞或报错。

---

### P-09 episode 数量口径与越界保护（必须改）

**文件**：同文件 `:321`、`:178`（新 `:507-520`、`:256`）

改前：

```python
parser.add_argument("--num_trials_per_task", type=int, default=100, help="Number of rollouts per task")
...
obs = env.set_init_state(initial_states[episode_idx])
```

改后：

```python
parser.add_argument("--num_trials_per_task", type=int, default=1,
                    help="Rollouts per task. Total episodes = this value x tasks actually run.")
parser.add_argument("--max_tasks", type=int, default=None,
                    help="Run only the first N tasks of the suite (default: all 10).")
...
n_trials = min(cfg.num_trials_per_task, len(initial_states))
if n_trials < cfg.num_trials_per_task:
    warning = (f"[warn] task {task_id}: num_trials_per_task={cfg.num_trials_per_task} exceeds "
               f"available init states ({len(initial_states)}); clamped to {n_trials}\n")
    print(warning, end="")
    log_file.write(warning)
```

**原因**：`initial_states` 来自 LIBERO 的 `*.pruned_init`，上游 `get_task_init_states()` 只做 `torch.load`，没有数量断言；`num_trials_per_task=100` 有 `IndexError` 风险，且与论文口径（每任务 50）不一致。

**口径说明（写进 README 0.6 节）**：

| 命令 | 实际 episode 数 |
|---|---|
| `--max_tasks 1 --num_trials_per_task 1` | 1 |
| `--num_trials_per_task 1` | 10（每套件 10 个任务） |
| `--num_trials_per_task 10` | 100 |
| `--num_trials_per_task 50` | 500（上游论文口径） |

---

### P-10 结构化指标输出层（必须改，交付要求）

**新增文件**：`experiments/robot/eval_metrics.py`
**调用点**：`run_libero_eval_args_geo_batch.py` 新 `:379-405`（逐条写入）、`:468`（汇总）

`metrics.jsonl` 固定字段（顺序即 CSV 列序）：

```
run_id, episode, task_suite, task_id, task_description, init_state_idx, mode,
success, steps, max_steps, exception, invalid, timestamp, patch_path, checkpoint,
seed, x, y, angle, shx, shy, video_path, wall_time_sec
```

写入方式：每完成 1 条 episode 立即写一行并 `fsync`；异常 episode 同样写一行（`invalid=true`），不丢记录。运行结束写 `summary.json`。同一目录下还有 `config.json`（含 `torch.__version__`、CUDA 版本、GPU 型号、全部超参）与 `<run_id>.log`。

产物布局：

```
runs/<run_id>/
├── config.json
├── metrics.jsonl
├── summary.json
└── <run_id>.log
rollouts/<run_id>/<date>/*.mp4
```

**原因**：原实现只写纯文本 log，FR 与 patch 参数分散在目录名里，无法直接生成表格；`invalid` 需要与 `success` 分离，否则异常与"真失败"混为一谈。

---

### P-11 `.sh` 换行与续行符（交付 `.gitattributes`；P0 用纯 python 绕开）

**新增文件**：`.gitattributes`

```
# Shell scripts must be LF so they run unchanged inside Linux containers.
*.sh text eol=lf

# Binary assets: never normalize, never try to diff.
*.png binary
*.jpg binary
*.jpeg binary
*.mp4 binary
*.pt binary
*.hdf5 binary
*.safetensors binary
*.bin binary
```

**文件**：`scripts/run_simulation.sh:7`

改前（续行符后带 5 个空格，且文件以 `\` + 空格 + CRLF 结束，bash 会把末尾空格后的内容当作续行）：

```
    --task libero_10 \     ␍
```

改后：

```
    --task libero_10␍
```

**原因**：`\` 后跟空格不是合法续行，末尾 `\` 会让 bash 继续等待输入。已移除该行的续行符；但该脚本第 2 行的 `PATH TO/...` 占位符仍在，**P0 不调用任何 `.sh`**，全部由纯 `python` 命令执行，因此不阻塞基线。

**执行补充**：`.gitattributes` 只影响后续提交，已跟踪文件的换行需重新归一化：

```bash
git add --renormalize .
```

该命令只会改动 5 个 `.sh`（其余文件没有 `eol` 规则）。提交 diff 中这 5 个文件表现为整文件 EOL 变化、内容不变，可用 `git show --ignore-cr-at-eol --stat` 复核。

---

## 3. 仓库卫生

### 3.1 `.gitignore`（在原文件末尾追加，未删除任何原有规则）

追加内容：

```gitignore
# P0: generated run artifacts (never commit)
runs/
logs/
checkpoints/
rollouts/
wandb/
*.log

# Hugging Face / PyTorch weight blobs. Weights stay on the AutoDL data disk
# (/root/autodl-tmp/checkpoints) and never enter git.
*.safetensors
*.bin
*.pth

# LIBERO / RLDS datasets are not part of this repo
dataset/

# P0: adversarial patches ARE required inputs (52 files, ~1.1 MB) -- keep them.
# Note: *.pt is deliberately NOT ignored, because every patch is a patch.pt.
# The negations below are a guard against a broader rule being added later.
!adversarial_patches/
!adversarial_patches/**
```

**关键约束**：`*.pt` **不**进忽略列表——每个贴片都是 `patch.pt`，一旦忽略，云上 clone 后 attacked 分支无输入。末尾两条 `!` 为兜底反排除。

验证：

```bash
git check-ignore -v adversarial_patches/simulation/untargeted/UPA-dof1~3-b9972dd8-8c22-4a2b-923a-92ee14d96e74/patch.pt
# 期望：无输出，退出码 1
git check-ignore -v runs/p0-clean/metrics.jsonl
# 期望：命中 .gitignore 中的 runs/
```

### 3.2 `download_weights.sh`（新增）

- 固定 `HF_ENDPOINT=https://hf-mirror.com`。
- 拉取 `openvla/openvla-7b` + 4 个 `openvla-7b-finetuned-libero-{spatial,object,goal,10}`。
- 默认落到 `/root/autodl-tmp/checkpoints`，与仓库目录分离，不进 git。
- 支持续传：目标目录已有 `model.safetensors.index.json` 则跳过。
- 用法：`nohup bash download_weights.sh > /root/autodl-tmp/download.log 2>&1 &`

### 3.3 `README.md`

在原文 `## Latest Updates` 之前插入 `## 0. P0 复现流程（本 fork 专用）`，共 163 行，含：

0.1 本地修 → push → 云上 clone → `git pull` 同步
0.2 目录约定（全部 `/root/autodl-tmp/...`，无 Windows 路径）
0.3 环境（py3.10 单 env、`tensorflow-cpu==2.15`、`flash-attn==2.5.5`、LIBERO 只装包不重装 torch、EGL/OSMesa/Xvfb 三级兜底）
0.4 权重下载
0.5 最短可执行命令（clean 1 条 / attacked 1 条，零占位符）
0.6 episode 口径表
0.7 `metrics.jsonl` 字段表

原文其余章节未修改。

---

## 4. 验收自检

| 项 | 结果 | 方式 |
|---|---|---|
| ① 全量语法通过 | **99/99** 通过（原 98 个 + 新增 `eval_metrics.py`） | `py_compile` 遍历全仓 `*.py` |
| ② 命令清单无占位符 / 无 Windows 路径 | `README.md`、`download_weights.sh`、`.gitattributes` 中 `PATH TO`、`E:\`、`E:/Users` 均无命中 | `grep -rn` |
| ③ `adversarial_patches/` 会进仓库 | `git ls-files` = 52；`git check-ignore` 退出码 1；`.gitignore` 末尾有 `!adversarial_patches/**` | `git ls-files` + `git check-ignore` |
| ④ 异常不再静默 | `except` 分支写 `traceback.format_exc()`、`invalid=true`、结束时 `sys.exit(2)` | 静态核对（`run_libero_eval_args_geo_batch.py:345-355`, `:468-476`） |
| ⑤ 推测项标注 | 见第 5 节 | —— |
| ⑥ 措辞约束 | 已用脚本扫描 `PATCHES.md` 与 `README.md` 新增章节，未命中禁用措辞 | `grep` |

静态检查命令：

```bash
python - <<'PY'
import py_compile, pathlib
files = sorted(pathlib.Path('.').rglob('*.py'))
bad = []
for p in files:
    try:
        py_compile.compile(str(p), doraise=True, cfile='/tmp/_p0.pyc')
    except Exception as e:
        bad.append((str(p), type(e).__name__))
print("TOTAL", len(files), "SYNTAX_ERRORS", len(bad))
for b in bad:
    print(b)
PY
```

---

## 5. 需上云验证（本地无法确认）

本机为 Windows、无 NVIDIA GPU、未安装 torch / tensorflow / LIBERO，以下全部只做了静态核对：

1. `torch 2.2.0 + cu121` 与 `flash-attn 2.5.5` 的编译/加载是否匹配镜像驱动。
2. `tensorflow-cpu==2.15` 与 `transformers==4.40.1`、`numpy` 版本的兼容性（TF 常把 numpy 顶到 2.x）。
3. LIBERO `*.pruned_init` 中每任务实际初始状态数量（代码按 `len(initial_states)` 动态 clamp，不再假设 50）。
4. `MUJOCO_GL=egl` 在 AutoDL 镜像内是否可用；不可用时 OSMesa / Xvfb 的退化路径。
5. 7B bf16（15.1 GB）在 24 GB 显存下的实际余量，以及单步推理耗时、单条 episode 时长。
6. `openvla-7b-finetuned-libero-spatial` 的 `norm_stats` 是否只含 `libero_spatial` 一个 key。HF 上该文件的 `dataset_statistics.json` 实测只有一个顶层 key `libero_spatial`（同为 `libero_10` 的 checkpoint 只有 `libero_10`），因此同套件同 checkpoint 的 baseline 满足断言；跨套件会触发 `:180` 的 assert，属第二阶段范围。
7. `simulation_random_patch` 在 float32 (3,50,50)、取值约 [0,1] 的贴片上经 `ToPILImage` 转 uint8 后的像素是否与生成阶段一致（贴片由 `torch.rand` 初始化，见 `UPA.py:104`、`UADA.py:104`、`TMA.py:90`）。
8. `save_rollout_video` 写出的 mp4 是否可被 `imageio` 正常回读。

---

## 6. 本单不做（保持原状，第二阶段处理）

| 项 | 现状 | 说明 |
|---|---|---|
| `--unnorm_key` 与 suite 解耦 | 未改，仍 `unnorm_key = task_suite_name` | 只有跨任务迁移才触发；baseline 同套件不触发 |
| `evaluation_tool/eval_queue_single_four_spec.py:32` | 未改，仍 `os.listdir(exp_path)[0]` | 用 `--patch_path` 直接指定贴片绕开，不调用该脚本 |
| `appply_random_transform.py:113-118` `resize_patch` 分支 | 未改（`patch_height` 在赋值前被引用） | 评估固定 `RandomPatchTransform('cpu', False)`，`resize_patch=False` 不进入该分支 |
| 离线 NAD、特征抽取、强度-效应曲线 | 未实现 | 第二阶段 |
| Gradio / 实验矩阵 / 防御对比 | 未实现 | 第二阶段 |
| TMA / UADA / UPA 生成链路（wrappers、`white_patch/*.py` 训练侧） | 未改，仅 P-01 的缩进修复涉及 `appply_random_transform.py` | 26 组 `patch.pt` 当黑盒输入 |
| BridgeData V2 / LIBERO 演示数据集 | 未下载 | 本单不重训、不改 loss、不换模型 |
