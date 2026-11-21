# Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics



[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

<div align="center">
  <img src=".\fig\mainfig.png">
</div>
<p>
Overall Adversarial Framework. 
</p>

Built on top of [OpenVLA](https://github.com/openvla/openvla), a remarkable generalist vision-language-action model work. 

---

## 0. P0 复现流程（本 fork 专用）

本 fork（`Randomlyu/vla-adv-patch-transfer`）在基线 commit `19d2e5a` 的基础上只做一件事：
让「clean 1 episode + attacked 1 episode」在单卡 4090 上跑通，并写出结构化指标。
逐条改动见 `PATCHES.md`。本机无 GPU，只做静态修复与语法校验，**所有运行时行为需上云验证**。

### 0.1 本地修 -> push -> 云上 clone -> 拉取同步

本地：

```bash
git status
git add -A
git commit -m "P0: minimal fixes for baseline evaluation"
git push origin main
```

云上（AutoDL 单卡 4090，Linux）：

```bash
mkdir -p /root/autodl-tmp
cd /root/autodl-tmp
git clone https://github.com/Randomlyu/vla-adv-patch-transfer.git
cd vla-adv-patch-transfer
git pull origin main          # 之后每次本地改完再执行这一行同步
```

### 0.2 目录约定（全部为 Linux 相对/绝对路径）

| 用途 | 路径 |
|---|---|
| 代码 | `/root/autodl-tmp/vla-adv-patch-transfer` |
| 模型权重 | `/root/autodl-tmp/checkpoints`（不进 git） |
| 运行产物 | `<repo>/runs/<run_id>/`（不进 git） |
| rollout 视频 | `<repo>/rollouts/<run_id>/<date>/*.mp4`（不进 git） |
| 对抗贴片 | `<repo>/adversarial_patches/`（在 git 内，勿删） |

### 0.3 环境（单 conda env，Python 3.10）

```bash
conda create -n vla-p0 python=3.10 -y
conda activate vla-p0

# 先确认镜像自带的 torch/CUDA，再决定是否重装（flash-attn 必须与 CUDA runtime 对齐）
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# 需要重装时（与 flash-attn 2.5.5 常用的 cu121 组合对齐）
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

cd /root/autodl-tmp/vla-adv-patch-transfer
pip install -e .                  # 按 pyproject.toml 拉取 pinned 依赖
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# 只需要 CPU 版 TF（图像 resize / center_crop 用到），避免与 torch 抢 CUDA 库
pip install tensorflow-cpu==2.15.0
```

LIBERO：只装包，不按它 README 重装 torch。

```bash
cd /root/autodl-tmp
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e . --no-deps
pip install robomimic hydra-core thop
pip install -r /root/autodl-tmp/vla-adv-patch-transfer/experiments/robot/libero/libero_requirements.txt
```

离屏渲染：先试 EGL，失败再退 OSMesa，最后退 Xvfb。

```bash
export MUJOCO_GL=egl
# 若报 "Renderer: egl not available"
#   export MUJOCO_GL=osmesa && apt-get update && apt-get install -y libosmesa6-dev
# 若两者都不可用
#   apt-get install -y xvfb && xvfb-run -a python ...
```

### 0.4 权重（第一天后台挂起，只写数据盘）

```bash
cd /root/autodl-tmp/vla-adv-patch-transfer
nohup bash download_weights.sh > /root/autodl-tmp/download.log 2>&1 &
tail -f /root/autodl-tmp/download.log
```

`download_weights.sh` 固定使用 `HF_ENDPOINT=https://hf-mirror.com`，拉取 `openvla/openvla-7b` 与 4 个
`openvla-7b-finetuned-libero-{spatial,object,goal,10}` 到 `/root/autodl-tmp/checkpoints`（约 76 GB）。

### 0.5 最短可执行命令（零占位符）

```bash
cd /root/autodl-tmp/vla-adv-patch-transfer
conda activate vla-p0
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

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

产物：

- `runs/p0-clean-spatial-t0/{config.json,metrics.jsonl,summary.json,<run_id>.log}`
- `rollouts/p0-clean-spatial-t0/<date>/<time>--episode=1--success=True--task=*.mp4`

### 0.6 episode 口径（避免误跑）

`--num_trials_per_task` 表示**每个任务**跑几条，不是总数。单次运行的总 episode 数 =
`num_trials_per_task x 实际任务数`；LIBERO 每个套件 10 个任务，`--max_tasks N` 只跑前 N 个。

| 命令 | 实际 episode 数 |
|---|---|
| `--max_tasks 1 --num_trials_per_task 1` | 1 |
| `--num_trials_per_task 1`（不加 max_tasks） | 10 |
| `--num_trials_per_task 10`（不加 max_tasks） | 100 |
| `--num_trials_per_task 50`（不加 max_tasks） | 500（上游论文口径） |

若 `num_trials_per_task` 超过该任务实际拥有的初始状态数，脚本会打印 `[warn] ... clamped`
并压到可用数量，不会 IndexError。

### 0.7 metrics.jsonl 字段

每次运行一个目录，`metrics.jsonl` 每完成一条 episode 追加一行 JSON（UTF-8，一行一对象），
`summary.json` 在运行结束时写一次汇总。后续表格、CSV 导出、分析界面只读这两个文件。

| 字段 | 含义 |
|---|---|
| `run_id` | 运行标识，同时是 `runs/` 下的目录名 |
| `episode` | 全局 episode 序号（从 1 开始） |
| `task_suite` / `task_id` / `task_description` | 套件名 / 任务序号 / 语言指令 |
| `init_state_idx` | 使用的初始状态下标（跨条件配对的主键） |
| `mode` | `clean` 或 `attacked` |
| `success` / `steps` / `max_steps` | 是否成功 / 实际执行步数（扣除 warm-up）/ 步数上限 |
| `exception` | `"类型: 信息"`，正常为 `null` |
| `invalid` | `true` 表示该条抛出异常，`success` 无意义 |
| `timestamp` / `wall_time_sec` | 完成时间 / 该条耗时（秒） |
| `patch_path` | 贴片绝对路径，clean 为 `null` |
| `checkpoint` / `seed` | 权重路径 / 随机种子 |
| `x` / `y` / `angle` / `shx` / `shy` | 贴片位置与几何参数 |
| `video_path` | 该条 rollout 的 mp4 路径，未产出帧时为 `null` |

任一条 episode 抛异常时，脚本打印完整 traceback、写 `invalid=true`，并在全部任务结束后
以退出码 `2` 退出。W&B 默认关闭，只有显式传 `--use_wandb` 才会上报，且需外部已配置 API key。

---

## Latest Updates
- [2025-03-08]
  - Upate UADA.
  - Bug fix.
  - Simple and flexible new evaluation tool (See in roboticAttack/evaluation_tool)
  - Release new ver. of Paper (ARXIV Release eta 11 Mar 2025 00:00:00 GMT).
  - ALL patches released (See in roboticAttack
/adversarial_patches).
- [2024-11-26] Pre release

---
## Important Resources
- Adversarial Patches (See adversarial_patches)
- Video (See videos)
---

## 1.Installation
(a) Use the setup commands below to get started:

```bash
conda create -n roboticAttack python=3.10 -y
conda activate roboticAttack

# Install PyTorch.
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Clone and install this repo
git clone https://github.com/William-wAng618/roboticAttack.git
cd roboticAttack
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

(b) Install LIBERO evaluation environment

```bash
# install LIBERO repo
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
# install other required packages:
cd ../ # cd roboticAttack/
pip install -r experiments/robot/libero/libero_requirements.txt
```

## 2.Dataset
We utilize two datasets for generating adversarial examples:

(a) BridgeData V2:\
Download the BridgeData V2 dataset:
```bash
# Change directory to your base datasets folder
cd roboticAttack

# Download the full dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# Rename the dataset to `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset datasets/bridge_orig
```

(b) LIBERO: \
Please download [this](https://huggingface.co/datasets/openvla/modified_libero_rlds/tree/main) preprocessed version of the LIBERO dataset, and place it in the `dataset/` folder.

(c) The structure should look like:

    ├── roboticAttack
    │   └── dataset
    |       └──bridge_orig
    |       └──libero_spatial_no_noops
    |       └──libero_object_no_noops
    |       └──libero_goal_no_noops
    |       └──libero_10_no_noops
## 2. Adversarial Patch Generation
(a) Target Manipulation Attack (TMA)
```bash
bash scripts/run_TMA.sh
```

(b) Untargeted Action Discrepancy (UADA)
```bash
bash scripts/run_UADA.sh
```

(b2) DDP ver For UADA
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 VLAAttacker/UADA_wrapper3_ddp.py \
    --maskidx 0 \
    --lr 1e-3 \
    --iter 2000 \
    --MSE_weights 5 \
    --accumulate 1 \                  
    --bs 8 \                            
    --warmup 20 \                        
    --tags XXX \                         
    --geometry True \                    
    --patch_size 3,50,50 \               
    --wandb_project your_wandb_project \ 
    --wandb_entity your_wandb_entity \   
    --innerLoop 50 \                     
    --dataset bridge_orig \              
    --resize_patch False \               
    --reverse_direction True           
```

(c) Untargeted Position-aware Attack (UPA)
```bash
bash scripts/run_UPA.sh
```

---
## 2.Evaluating OpenVLA

```bash
bash scripts/run_simulation.sh
```

---

## Repository Structure

High-level overview of repository/project file-tree:

+ `VLAAttcker/` - Including the code for generating adversarial examples (UADA, UPA, TMA).
+ `scripts/` - Scripts for Attack and Simulation.
+ `experiments/` - Code for evaluating OpenVLA policies in robot environments.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!

---


## 📚 Citation

If you find this project useful, please consider citing our paper:

```bibtex
@InProceedings{Wang_2025_ICCV,
    author    = {Wang, Taowen and Han, Cheng and Liang, James and Yang, Wenhao and Liu, Dongfang and Zhang, Luna Xinyu and Wang, Qifan and Luo, Jiebo and Tang, Ruixiang},
    title     = {Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {6948--6958}
}

