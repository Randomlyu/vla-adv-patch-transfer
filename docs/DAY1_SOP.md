# 上云首日 SOP（P0 基线）

> 适用：AutoDL 按量 RTX 4090 单卡，公开仓库 HTTPS clone，Python 3.10 单 conda env。
> 配套：`docs/ERROR_PLAYBOOK.md`（10 条报错预案）、`PATCHES.md`（改动清单）、`README.md` §0（复现流程）。
> 本文件只覆盖「从租卡到拿到第一条 clean + attacked rollout」，不含实验设计。
> 时间基准：距 12/18 开题答辩约 4 周。

---

## 0. 租卡与计费（一页表）

### 0.1 平台结论

AutoDL 按量 4090 仍然是当前这套任务的首选：单卡 24 GB 足够 7B bf16 推理，国内下载 `hf-mirror` 速度快，镜像自带 CUDA、无需自建。

| 机型 | 显存 | 参考单价（¥/h） | 与本任务的适配性 | 何时才用 |
|---|---|---|---|---|
| **RTX 4090** | 24 GB | 官网 1.88 / 第三方比价 1.8–2.8 | 目标卡，sm89，bf16 + flash-attn 2.x 可用 | **默认** |
| RTX 3090 | 24 GB | 1.32 | sm86，bf16 + FA2 可用；单步比 4090 慢约 30–40% | 4090 无卡可租时的**同架构**备选 |
| RTX 5090 | 32 GB | 2.78 | sm120，`flash-attn==2.5.5` 是否已有对应编译需现场验证 | 只在 4090/3090 都无卡时考虑，先跑 D-01 验证 |
| A100-40G / 80G | 40 / 80 GB | 4.5–6.5（第三方）/ 官网 A800-80G 5.59 | sm80，FA2 原生支持，显存余量大 | 反复 OOM 或需要同时驻留模型与更大缓存时 |
| **V100** | 32 GB | 1.88 | **不适用** | 不使用 |

**V100 为什么不能当兜底**

1. V100 是 Volta（sm70），**没有原生 bf16**；而 `experiments/robot/openvla_utils.py:46` 固定 `torch_dtype=torch.bfloat16`。在 V100 上要么报错，要么静默退化成 fp16/fp32。
2. `flash-attn` 2.x 只支持 Ampere 及更新架构（sm80+）；`openvla_utils.py:45` 固定 `attn_implementation="flash_attention_2"`，V100 上直接不可用。
3. 所以 V100 不是"便宜一点的同一台机器"，而是**换了一套数值行为**。用它跑出来的 SR 不能和 4090 上的结果放在同一张表里。

> 单价为公开页面参考值，**以控制台现价为准**；不同地区、不同时段价格会浮动。

### 0.2 实例选择

| 项 | 结论 |
|---|---|
| 机型版本 | 避开"青春版 / 迷你版"一类变体：通常不支持无卡开机、不支持保存镜像（以控制台机型说明为准） |
| 镜像 | PyTorch 2.x + CUDA 12.x + Python 3.10；与 `pyproject.toml` 的 `torch==2.2.0` 对齐最省事 |
| 数据盘 50 GB | **够跑 baseline**：1 个 checkpoint 15.1 GB + 余量；**不够全量**：5 个仓库约 75 GB，需扩到 ≥150 GB |
| 系统盘 30 GB | 装 conda 环境（torch + TF + flash-attn 约 10–15 GB）偏紧；若 `/` 使用率超 85%，把 conda 环境迁到数据盘 |

### 0.3 首日模式切换

| 阶段 | 模式 | 计费 | 做什么 |
|---|---|---|---|
| ① | **无卡模式** | ¥0.1/h | clone、建 conda、装包、下权重（CPU 下载）、静态自检（D-03 / D-09 / D-10 探针） |
| ② | **有卡模式** | 1.88 起 | 跑第 1 条 clean、第 1 条 attacked |

**为什么不要反复切**

- 每次切换要关机再开机，有 1–2 分钟启动，热门机型还可能排队。
- 切换会丢失当前 shell 的环境变量（`MUJOCO_GL`、`CUDA_VISIBLE_DEVICES`、`HF_ENDPOINT`），每次都要重设，容易漏。
- 无卡模式下 CUDA 相关报错（D-01 / D-05）根本复现不了，在无卡模式里排查显存问题是空转。
- 结论：**只在"包装完 + 权重下完 + 静态自检通过"之后切一次**，见 §4 的唯一切换点。

**计费要点**

- 开机即计费：跑脚本、下数据、挂着不动都算。
- 关机停止算力计费，但**扩容的数据盘按日计费，关机也照收**。
- 连续关机超过 15 天实例可能被回收（以控制台公告为准）；权重与实验结果需要额外备份到网盘 / 对象存储，不能只放在数据盘。
- 学生认证与会员折扣以控制台活动页为准（官网标注会员 95 折）。

### 0.4 分段预算

| 阶段 | GPU 小时 | 单价（¥/h） | 区间（¥） | 备注 |
|---|---|---|---|---|
| 无卡配环境 + 下权重 | 2–4 h | 0.1 | 0.2–0.4 | 主要是下载等待 |
| 首日有卡调试 | 3–6 h | 1.9–2.8 | 6–17 | 2 条 rollout + 排错 |
| 开题前预研 | 20–40 h | 1.9–2.8 | 38–112 | 少量条件试跑 |
| 论文阶段全量 | 60–120 h | 1.9–2.8 | 114–336 | **单步耗时需上云实测校准** |
| 合计 | | | **160–470** | 不含数据盘扩容 |

> 全部为估算区间；**单步耗时必须用首日实测值替换**（`metrics.jsonl` 的 `wall_time_sec` 除以 `steps`）。**价格以控制台现价为准**。

---

## 1. 公开仓库合规核查

### 1.1 LICENSE

| 项 | 实测结论 |
|---|---|
| 是否存在 | 存在，仓库根 `LICENSE` |
| 许可类型 | **MIT License** |
| 版权行 | `Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti.`（即 OpenVLA 作者，不是 roboticAttack 作者） |
| 公开再分发需保留 | MIT 要求**保留版权声明与许可声明原文**，即在分发时继续附带该 `LICENSE`；同时保留"修改过"的说明（本仓库的修改说明即 `PATCHES.md`） |
| 现状判断 | `LICENSE` 未被本 fork 改动，满足保留要求；但因为版权行指向 OpenVLA，仓库需要一段来源说明把"OpenVLA → roboticAttack → 本 fork"这条链讲清楚，见 1.2 |

### 1.2 README 署名现状与应补文本

现状（实测）：

- README 有 `Built on top of [OpenVLA]` 一句（第 16 行附近）；
- README 有 `## 📚 Citation` 段，含 ICCV 2025 的 `@InProceedings{Wang_2025_ICCV, ...}`；
- README **没有**"本仓库基于 `William-wAng618/roboticAttack` commit `a0bef50`"的说明；
- README 仍保留上游 `git clone https://github.com/William-wAng618/roboticAttack.git` 的安装命令（第 210 行），与当前 fork 的实际 clone 地址不一致。

应补文本（**仅给出文本，本单未改动 README**）：

```markdown
## Attribution

This repository is a research fork of
[William-wAng618/roboticAttack](https://github.com/William-wAng618/roboticAttack)
at commit `a0bef50`, which accompanies:

> Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics.
> Wang et al., ICCV 2025.

The upstream project is built on [OpenVLA](https://github.com/openvla/openvla) and is
distributed under the MIT License (see `LICENSE`, Copyright (c) 2024 Moo Jin Kim,
Karl Pertsch, Siddharth Karamcheti). This fork keeps the original `LICENSE` unchanged;
the evaluation-pipeline changes are documented in `PATCHES.md`.

This repository is intended for security research on embodied AI. Do not use it on
hardware or in environments where a failed policy could cause injury or property damage.
```

### 1.3 转 Public 前检查清单

- [ ] `LICENSE` 保留原文、版权行未被改动
- [ ] README 加上 §1.2 的 Attribution 段（含 commit `a0bef50` 与 ICCV 2025 引用）
- [ ] README 第 210 行的 clone 地址改为本 fork 地址（避免访客拉到上游）
- [ ] `git log --format='%an <%ae>' | sort -u` 检查提交者邮箱是否会随公开暴露（当前为个人邮箱；不接受则先改 GitHub 邮箱隐私设置，再决定是否重写历史）
- [ ] `.gitignore` 已覆盖 `.env` / `*.env` / `*token*` / `*.key` / `*.pem` / `rollouts/`（本单已改，见交付说明）
- [ ] 全仓无明文凭据：`grep -rnEi "hf_[a-z0-9]{10,}|sk-[a-z0-9]{10,}|api[_-]?key|password" --include="*.py" --include="*.sh" --include="*.md" .`
- [ ] 无单文件超过 100 MB（GitHub 硬限制）：`git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '$3>104857600'`
- [ ] `videos/` 与 `adversarial_patches/` 确认是上游随仓库发布的素材，可再分发
- [ ] 仓库 Description 写明"security research / adversarial patches for VLA"，避免被误读为通用工具
- [ ] 确认没有 `.github/workflows` 里的 secret 引用（本仓库无该目录）
- [ ] 转 Public 后立刻用未登录浏览器打开一次，确认无 401、无私有链接

---

## 2. 报错预案

见 **`docs/ERROR_PLAYBOOK.md`**，共 10 条：`PATCHES.md` 第 5 节的 8 条 + `parents[3]` 探针 + `cuda:0` 与 `CUDA_VISIBLE_DEVICES` 专项。每条含典型症状、一行根因、可粘贴修法、兜底方案。

---

## 3. 日志回传协议

云上出问题时，**一次性**把下列内容交回，避免来回追问。

### 3.1 必传清单

| # | 内容 | 为什么必须 |
|---|---|---|
| 1 | 完整 traceback 原文（不是最后一行） | 定位到具体文件行 |
| 2 | 执行命令原文（含所有参数） | 排除参数错配 |
| 3 | `nvidia-smi` 输出 | 显存占用、残留进程、驱动版本 |
| 4 | torch 版本与 CUDA 可用性 | D-01 / D-05 的第一判据 |
| 5 | `pip freeze` 中 torch / flash-attn / tensorflow / transformers / numpy | ABI 与依赖冲突 |
| 6 | `MUJOCO_GL` / `CUDA_VISIBLE_DEVICES` / `HF_ENDPOINT` 的取值 | D-04 / D-10 / 下载源 |
| 7 | `runs/<run_id>/config.json` | 记录了 GPU 型号、torch/CUDA 版本、全部超参 |
| 8 | `runs/<run_id>/metrics.jsonl` 最后一行 | 是否 `invalid=true`、异常类型 |
| 9 | `rollouts/` 目录清单 | 是否真的产出了 mp4 |

### 3.2 运行时先落盘（每次运行都这样做）

```bash
mkdir -p logs
CMD='python experiments/robot/libero/run_libero_eval_args_geo_batch.py --pretrained_checkpoint /root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial --task_suite_name libero_spatial --max_tasks 1 --num_trials_per_task 1 --clean --run_id p0-clean-spatial-t0'
printf '%s\n' "$CMD" > logs/last_cmd.txt
eval "$CMD" 2>&1 | tee "logs/last_run.log"
```

跑 attacked 时把 `CMD` 换成 README §0.5 第 2 条那行即可，其余不变。

### 3.3 出问题后一键收集

```bash
mkdir -p logs && {
  echo "===== 1. time / host ====="; date -Is; hostname
  echo "===== 2. command ====="; cat logs/last_cmd.txt 2>/dev/null || echo "(logs/last_cmd.txt missing)"
  echo "===== 3. git ====="; git log --oneline -3; git status --short
  echo "===== 4. nvidia-smi ====="; nvidia-smi
  echo "===== 5. torch ====="; python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'avail=', torch.cuda.is_available(), 'count=', torch.cuda.device_count()); print('name0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
  echo "===== 6. env ====="; env | grep -E "^(MUJOCO_GL|CUDA_VISIBLE_DEVICES|HF_ENDPOINT|PYTORCH_CUDA_ALLOC_CONF)=" || true
  echo "===== 7. pip freeze (filtered) ====="; pip freeze | grep -iE "^(torch|torchvision|torchaudio|flash[-_]attn|tensorflow|transformers|numpy|tokenizers|timm|accelerate|imageio|libero|robosuite|mujoco|bddl)==" || true
  echo "===== 8. disk ====="; df -h / /root/autodl-tmp
  echo "===== 9. runs tree ====="; find runs -maxdepth 2 -type f 2>/dev/null | head -40
  echo "===== 10. config.json ====="; cat runs/*/config.json 2>/dev/null | head -80
  echo "===== 11. metrics.jsonl last line ====="; tail -n 1 runs/*/metrics.jsonl 2>/dev/null
  echo "===== 12. rollouts ====="; find rollouts -type f -name "*.mp4" 2>/dev/null | head -20
  echo "===== 13. run log tail ====="; tail -n 120 logs/last_run.log 2>/dev/null
} 2>&1 | tee logs/first_run.log
```

`logs/` 已在 `.gitignore` 中，不会被推到公开仓库。回传时把 `logs/first_run.log` 全文贴回即可（内含 traceback 与全部环境信息）。

---

## 4. 首日时间线

### 4.1 步骤与并行关系

| 步 | 动作 | 模式 | 并行关系 | 产物 |
|---|---|---|---|---|
| S1 | `git clone` 公开仓库（HTTPS） | 无卡 | 可与 S2 并行 | 仓库目录 |
| S2 | `conda create` + `pip install -e .` + `flash-attn` + `tensorflow-cpu` | 无卡 | **与 S3 并行**（同为网络 IO） | 可用 env |
| S3 | `nohup bash download_weights.sh > /root/autodl-tmp/download.log 2>&1 &` | 无卡 | **与 S2 并行** | 15.1 GB checkpoint |
| S4 | 静态自检：`py_compile` 全量、D-03 / D-09 / D-10 探针 | 无卡 | 串行，必须在 S2 之后 | 自检输出 |
| S5 | 切有卡（**唯一切换点**，见 4.3） | 有卡 | —— | —— |
| S6 | README §0.5 第 1 条：clean 1 条 | 有卡 | 串行 | `rollouts/p0-clean-spatial-t0/*.mp4` |
| S7 | README §0.5 第 2 条：attacked 1 条 | 有卡 | 串行（换贴片路径与 run_id） | `rollouts/p0-atk-spatial-t0/*.mp4` |
| S8 | §5 成功判据逐条勾选 + §3.3 收集日志 | 有卡 | 串行 | `logs/first_run.log` |

**可并行**：S2（装包）∥ S3（下权重）——两者都吃带宽，实际会比串行慢一些，但总时长仍短于串行；不要让两者同时占满磁盘 IO。
**必须串行**：S2 → S4 → S5 → S6 → S7 → S8。

### 4.2 引用的两条命令（摘自 README §0.5，未改动）

```bash
# 1) clean，恰好 1 条 episode
python experiments/robot/libero/run_libero_eval_args_geo_batch.py \
  --pretrained_checkpoint /root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --max_tasks 1 --num_trials_per_task 1 \
  --clean \
  --run_id p0-clean-spatial-t0

# 2) attacked，恰好 1 条 episode
python experiments/robot/libero/run_libero_eval_args_geo_batch.py \
  --pretrained_checkpoint /root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --max_tasks 1 --num_trials_per_task 1 \
  --patch_path "adversarial_patches/simulation/untargeted/UPA-dof1~3-b9972dd8-8c22-4a2b-923a-92ee14d96e74/patch.pt" \
  --x 120 --y 160 \
  --run_id p0-atk-spatial-t0
```

### 4.3 权重没下完时能否先跑

**结论：不能跑 rollout。** 两条命令都要加载 15.1 GB 的 checkpoint，权重不全时 `from_pretrained` 会直接失败。

可以在等待期间并行做的事：

```bash
python -m compileall -q experiments prismatic VLAAttacker && echo "compile ok"
python - <<'PY'
from pathlib import Path
f = Path("experiments/robot/libero/run_libero_eval_args_geo_batch.py").resolve()
print("parents[3] =", f.parents[3], "| white_patch exists:",
      (f.parents[3] / "VLAAttacker" / "white_patch" / "appply_random_transform.py").is_file())
PY
```

判断权重是否下完（唯一标准）：

```bash
ls -1 /root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial/model-0000*-of-00004.safetensors | wc -l
# 期望 4；再确认 index 与 stats 都在
ls -l /root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial/{model.safetensors.index.json,dataset_statistics.json}
du -sh /root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial
# 期望约 15–16G
```

### 4.4 无卡 → 有卡的唯一切换点

同时满足这三条才切：

1. `4` 个 `model-0000*-of-00004.safetensors` 全部存在，且 `model.safetensors.index.json`、`dataset_statistics.json` 存在；
2. `python -c "import torch, tensorflow; print(torch.__version__, tensorflow.__version__)"` 无报错；
3. `python -m compileall -q experiments prismatic VLAAttacker` 通过。

不满足任一条就切有卡，等于花 1.88 ¥/h 在调试环境。

---

## 5. 成功判据（手动勾选）

> 以下路径以 clean 那条为例；attacked 把 `p0-clean-spatial-t0` 换成 `p0-atk-spatial-t0`。

- [ ] **mp4 存在**
  ```bash
  ls -lh rollouts/p0-clean-spatial-t0/*/*.mp4
  ```
- [ ] **mp4 可回读且帧数 > 1**
  ```bash
  python -c "import glob, imageio.v2 as iio; p=glob.glob('rollouts/p0-clean-spatial-t0/*/*.mp4')[0]; r=iio.get_reader(p); print(p, sum(1 for _ in r))"
  ```
- [ ] **`metrics.jsonl` 恰好 1 行且 `invalid=false`**
  ```bash
  python -c "import json; rows=[json.loads(l) for l in open('runs/p0-clean-spatial-t0/metrics.jsonl')]; print('rows=',len(rows), 'invalid=',rows[0]['invalid'], 'success=',rows[0]['success'], 'steps=',rows[0]['steps'])"
  ```
- [ ] **`summary.json` 字段齐全**
  ```bash
  python -c "import json; d=json.load(open('runs/p0-clean-spatial-t0/summary.json')); print(sorted(d))"
  ```
  期望含：`run_id, mode, task_suite, checkpoint, patch_path, x, y, angle, shx, shy, seed, started_at, finished_at, tasks_run, num_trials_per_task, total_episodes, total_successes, total_success_rate, invalid_episodes, episode_stats`
- [ ] **`config.json` 记录到 GPU 型号与版本**
  ```bash
  python -c "import json; d=json.load(open('runs/p0-clean-spatial-t0/config.json')); print(d['cuda_device_name'], '|', d['torch_version'], '| cuda', d['torch_cuda'])"
  ```
  期望形如 `NVIDIA GeForce RTX 4090 | 2.2.0 | cuda 12.1`
- [ ] **attacked 与 clean 的差异可核对**（同一 `task_id` / `init_state_idx`）
  ```bash
  python -c "import json; a=[json.loads(l) for l in open('runs/p0-clean-spatial-t0/metrics.jsonl')]; b=[json.loads(l) for l in open('runs/p0-atk-spatial-t0/metrics.jsonl')]; print('clean', a[0]['success'], a[0]['steps']); print('atk  ', b[0]['success'], b[0]['steps'], b[0]['patch_path'])"
  ```
  两条的 `task_id` 与 `init_state_idx` 必须相同，`patch_path` 必须非空。
- [ ] **进程退出码为 0**
  ```bash
  echo "exit_code=$?"
  ```

### 5.1 出现这些情况必须停下来贴日志，不要重跑

| 情况 | 判定信号 | 为什么不能重跑 |
|---|---|---|
| 同一命令连续 2 次相同 traceback | `logs/first_run.log` 里有两份一模一样的栈 | 是确定性错误，重跑 100 次也一样，只会烧钱 |
| `CUDA out of memory` | D-05 症状 | 重跑会让残留进程叠加，显存越跑越少 |
| `metrics.jsonl` 出现 `invalid=true` | 该行 `exception` 非空 | 这是一条"假失败"，会把 FR 算低；先修根因 |
| attacked 与 clean 的 `success`/`steps` 完全相同 | §5 最后一条命令输出相等 | 贴片可能没生效（D-07），此时的数据不可用 |
| mp4 帧数为 0 或文件 0 字节 | D-08 症状 | 交付物缺失，且通常伴随上游异常 |
| 下载中断后 `du -sh` 明显小于 15 G | 4.3 的检查 | 半包权重会让 `from_pretrained` 报形状错，误判成代码问题 |
| 连续两次切换无卡/有卡 | 计费与排队成本 | 先确认失败是否与 CUDA 有关；无关就别切 |

---

## 6. 证据 ↔ 开题报告章节映射（只做映射）

| 证据 | 对应章节 | 呈现形式 | 使用前提 |
|---|---|---|---|
| `rollouts/p0-clean-spatial-t0/*.mp4` | 技术路线 —— 数据流「图像获取 → 贴片叠加 → OpenVLA 推理 → action → LIBERO 步进」 | 1 张关键帧截图，贴在流程图对应节点旁 | 证明链路可运行，不证明效果 |
| `rollouts/p0-atk-spatial-t0/*.mp4` | 初步实验结果 —— 攻击生效的定性证据 | 与 clean **同 `init_state_idx`** 的并排关键帧（左右对比） | 两条视频必须来自同一个初始状态 |
| `metrics.jsonl`（clean + attacked 各 1 行） | 初步实验结果 —— 首条可量化记录 | 两行三列的表格：`success` / `steps` / `patch_path` | 只有 1 条，**不做统计推断** |
| `metrics.jsonl` 的 `wall_time_sec` 与 `steps` | 可行性分析 —— 单条 rollout 时延与算力预算 | 一行结论 + 换算成"每 GPU 小时可跑多少条" | 需上云实测，替换预算表里的估算值 |
| `summary.json` | 可行性分析 —— `invalid_episodes=0` 说明链路稳定 | 一行结论 | 与 metrics 一致 |
| `config.json` | 技术路线 —— 环境可复现（GPU 型号、torch/CUDA 版本、全部超参） | 截图（含 `cuda_device_name` / `torch_version`） | 版本锁定证据 |
| `logs/first_run.log`（§3.3 收集） | 附录 —— 复现证据 | 代码块（截取前 60 行与环境段） | 含命令原文与 traceback |
| `PATCHES.md` 的 P-01…P-11 | 技术路线 —— 工程改动与可运行性论证 | 表格（不改写，直接引用） | 说明"为什么原仓库跑不起来" |

> 时间基准：距 12/18 开题答辩约 4 周；上述证据只需在开题前各拿到 1 份，不做多条件扩展。
