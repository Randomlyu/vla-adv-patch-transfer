"""
run_libero_eval_args_geo_batch.py

Runs an OpenVLA checkpoint in a LIBERO simulation environment, optionally pasting a
pre-generated adversarial patch into the agent view.

P0 baseline usage (see PATCHES.md / README.md):

    # clean episode
    python experiments/robot/libero/run_libero_eval_args_geo_batch.py \
        --pretrained_checkpoint /root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial --max_tasks 1 --num_trials_per_task 1 --clean

    # attacked episode
    python experiments/robot/libero/run_libero_eval_args_geo_batch.py \
        --pretrained_checkpoint /root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial --max_tasks 1 --num_trials_per_task 1 \
        --patch_path adversarial_patches/simulation/untargeted/UPA-dof1~3-b9972dd8-8c22-4a2b-923a-92ee14d96e74/patch.pt

Episode accounting: total rollouts = num_trials_per_task * number_of_tasks actually run
(10 per LIBERO suite unless --max_tasks is given).
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import tqdm
from libero.libero import benchmark

import wandb

# Repo root = three levels up from <repo>/experiments/robot/libero/this_file.py
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "VLAAttacker" / "white_patch"))

from appply_random_transform import RandomPatchTransform  # noqa: E402

from experiments.robot.eval_metrics import MetricsWriter, aggregate  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# Longest training demo per suite (kept identical to upstream values).
MAX_STEPS = {
    "libero_spatial": 193,
    "libero_object": 254,
    "libero_goal": 270,
    "libero_10": 505,
    "libero_90": 373,
}


def _safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


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


def _build_run_id(cfg, mode):
    if cfg.run_id:
        return cfg.run_id
    note = cfg.run_id_note or "nonote"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"EVAL-{cfg.task_suite_name}--{mode}--{note}--{stamp}"


# @dataclass
# class GenerateConfig:
#     # fmt: off
#
#     #################################################################################################################
#     # Model-specific parameters
#     #################################################################################################################
#     model_family: str = "openvla"                    # Model family
#     pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
#     load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
#     load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
#
#     center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
#
#     #################################################################################################################
#     # LIBERO environment-specific parameters
#     #################################################################################################################
#     task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
#     num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
#     num_trials_per_task: int = 50                    # Number of rollouts per task
#
#     #################################################################################################################
#     # Utils
#     #################################################################################################################
#     run_id_note: Optional[str] = "spatial1"                # Extra note to add in run ID for logging
#     local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
#
#     use_wandb: bool = False                          # Whether to also log results in Weights & Biases
#     wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
#     wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
#
#     seed: int = 7                                    # Random Seed (for reproducibility)
#
#     # fmt: on


def eval_libero(cfg) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # ------------------------------------------------------------------ patch / mode
    patch_path = None if cfg.clean else _resolve_patch_path(cfg.patch_path or cfg.patchroot)
    mode = "clean" if patch_path is None else "attacked"
    patch = _load_patch(patch_path) if patch_path is not None else None

    randomPatchTransform = RandomPatchTransform("cpu", False)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    # NOTE: kept equal to the suite name on purpose. Detaching it from the suite (needed for
    # cross-suite transfer) is out of scope for P0 and would change the action scale.
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, (
            f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`! "
            f"available keys: {list(model.norm_stats.keys())}"
        )

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # ------------------------------------------------------------------ run directory
    run_id = _build_run_id(cfg, mode)
    cfg.run_id = run_id
    cfg.mode = mode
    cfg.patch_path_resolved = None if patch_path is None else str(patch_path)
    cfg.exp_name = run_id

    run_dir = Path(cfg.metrics_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = dict(vars(cfg))
    config_snapshot["torch_version"] = torch.__version__
    config_snapshot["torch_cuda"] = torch.version.cuda
    config_snapshot["cuda_device_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    (run_dir / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    metrics = MetricsWriter(run_dir / "metrics.jsonl")

    local_log_filepath = run_dir / f"{run_id}.log"
    log_file = open(local_log_filepath, "w", encoding="utf-8")
    print(f"Logging to local log file: {local_log_filepath}")
    print(f"metrics.jsonl: {run_dir / 'metrics.jsonl'}")
    print(f"mode={mode} patch={cfg.patch_path_resolved}")

    # Initialize Weights & Biases logging as well (off unless --use_wandb is passed)
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    max_steps = MAX_STEPS[cfg.task_suite_name]

    tasks_to_run = num_tasks_in_suite if not cfg.max_tasks else min(cfg.max_tasks, num_tasks_in_suite)
    print(f"tasks_to_run={tasks_to_run} num_trials_per_task={cfg.num_trials_per_task} "
          f"-> planned episodes={tasks_to_run * cfg.num_trials_per_task}")

    # Start evaluation
    total_episodes, total_successes, invalid_episodes = 0, 0, 0
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    for task_id in tqdm.tqdm(range(tasks_to_run)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # `initial_states[episode_idx]` indexes the shipped *.pruned_init tensors; clamp so an
        # oversized --num_trials_per_task cannot raise IndexError.
        n_trials = min(cfg.num_trials_per_task, len(initial_states))
        if n_trials < cfg.num_trials_per_task:
            warning = (
                f"[warn] task {task_id}: num_trials_per_task={cfg.num_trials_per_task} exceeds "
                f"available init states ({len(initial_states)}); clamped to {n_trials}\n"
            )
            print(warning, end="")
            log_file.write(warning)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(n_trials)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []
            exception = None
            invalid = False
            episode_started = time.time()

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            try:
                while t < max_steps + cfg.num_steps_wait:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    if patch is not None:
                        img = randomPatchTransform.simulation_random_patch(
                            img,
                            patch,
                            geometry=True,
                            colorjitter=False,
                            angle=cfg.angle,
                            shx=cfg.shx,
                            shy=cfg.shy,
                            position=(cfg.x, cfg.y),
                        )
                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        break
                    t += 1

            except Exception as err:  # noqa: BLE001 - every failure must be recorded, not swallowed
                invalid = True
                exception = f"{type(err).__name__}: {err}"
                traceback_text = traceback.format_exc()
                print(traceback_text, file=sys.stderr)
                log_file.write(traceback_text)
                log_file.flush()

            if done and not invalid:
                task_successes += 1
                total_successes += 1

            task_episodes += 1
            total_episodes += 1
            if invalid:
                invalid_episodes += 1

            # Save a replay video of the episode
            video_path = None
            if replay_images:
                # Append the frame from the final env.step() so the video ends on the terminal state
                img = get_libero_image(obs, resize_size)
                if patch is not None:
                    img = randomPatchTransform.simulation_random_patch(
                        img,
                        patch,
                        geometry=True,
                        colorjitter=False,
                        angle=cfg.angle,
                        shx=cfg.shx,
                        shy=cfg.shy,
                        position=(cfg.x, cfg.y),
                    )
                replay_images.append(img)
                print("Saving replay video...")
                video_path = save_rollout_video(
                    replay_images,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    log_file=log_file,
                    exp_name=cfg.exp_name,
                )
            else:
                print("[warn] no frames captured; replay video skipped")
                log_file.write("[warn] no frames captured; replay video skipped\n")

            # Structured per-episode record
            metrics.write_episode(
                {
                    "run_id": run_id,
                    "episode": total_episodes,
                    "task_suite": cfg.task_suite_name,
                    "task_id": task_id,
                    "task_description": task_description,
                    "init_state_idx": episode_idx,
                    "mode": mode,
                    "success": bool(done and not invalid),
                    "steps": max(t - cfg.num_steps_wait, 0),
                    "max_steps": max_steps,
                    "exception": exception,
                    "invalid": invalid,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "patch_path": cfg.patch_path_resolved,
                    "checkpoint": cfg.pretrained_checkpoint,
                    "seed": cfg.seed,
                    "x": cfg.x,
                    "y": cfg.y,
                    "angle": cfg.angle,
                    "shx": cfg.shx,
                    "shy": cfg.shy,
                    "video_path": video_path,
                    "wall_time_sec": round(time.time() - episode_started, 3),
                }
            )

            # Log current results
            print(f"Success: {done and not invalid}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({(_safe_div(total_successes, total_episodes) or 0) * 100:.1f}%)")
            log_file.write(f"Success: {done and not invalid}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({(_safe_div(total_successes, total_episodes) or 0) * 100:.1f}%)\n"
            )
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {_safe_div(task_successes, task_episodes)}")
        print(f"Current total success rate: {_safe_div(total_successes, total_episodes)}")
        log_file.write(f"Current task success rate: {_safe_div(task_successes, task_episodes)}\n")
        log_file.write(f"Current total success rate: {_safe_div(total_successes, total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb and task_episodes:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    total_success_rate = _safe_div(total_successes, total_episodes)
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": total_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(str(local_log_filepath))

    summary = {
        "run_id": run_id,
        "mode": mode,
        "task_suite": cfg.task_suite_name,
        "checkpoint": cfg.pretrained_checkpoint,
        "patch_path": cfg.patch_path_resolved,
        "x": cfg.x,
        "y": cfg.y,
        "angle": cfg.angle,
        "shx": cfg.shx,
        "shy": cfg.shy,
        "seed": cfg.seed,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks_run": tasks_to_run,
        "num_trials_per_task": cfg.num_trials_per_task,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "invalid_episodes": invalid_episodes,
    }
    summary["episode_stats"] = aggregate(metrics.records)
    metrics.close(summary)

    if invalid_episodes:
        print(
            f"[fatal] {invalid_episodes}/{total_episodes} episode(s) raised exceptions; "
            f"rows with invalid=true in {run_dir / 'metrics.jsonl'}",
            file=sys.stderr,
        )
        sys.exit(2)


def parse_args():
    parser = argparse.ArgumentParser(description="LIBERO evaluation with an optional adversarial patch (P0)")
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    parser.add_argument("--model_family", type=str, default="openvla", help="Model family")
    parser.add_argument("--exp_name", type=str, default="p0", help="Rollout folder name (overridden by run_id)")
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="openvla/openvla-7b-finetuned-libero-object",
        help="Pretrained checkpoint path (local dir or HF repo id)",
    )
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--center_crop", dest="center_crop", action="store_true", default=True)
    parser.add_argument("--no_center_crop", dest="center_crop", action="store_false")

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_object",
        help="Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90",
    )
    parser.add_argument("--num_steps_wait", type=int, default=10, help="Steps to wait for objects to stabilize")
    parser.add_argument(
        "--num_trials_per_task",
        type=int,
        default=1,
        help="Rollouts per task. Total episodes = this value x tasks actually run.",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Run only the first N tasks of the suite (default: all 10).",
    )

    #################################################################################################################
    # Patch parameters
    #################################################################################################################
    parser.add_argument(
        "--patch_path",
        type=str,
        default=None,
        help="Path to a pre-generated patch.pt. Omit (or pass 'none') for a clean run.",
    )
    parser.add_argument("--clean", action="store_true", help="Force a clean run; ignores --patch_path")
    parser.add_argument("--patchroot", type=str, default=None, help="Deprecated alias of --patch_path")
    parser.add_argument("--x", type=int, default=5, help="Patch paste column in the 224x224 agent view")
    parser.add_argument("--y", type=int, default=160, help="Patch paste row in the 224x224 agent view")
    parser.add_argument("--angle", type=float, default=0.0, help="Patch rotation in degrees")
    parser.add_argument("--shx", type=float, default=0.0, help="Patch shear x")
    parser.add_argument("--shy", type=float, default=0.0, help="Patch shear y")

    #################################################################################################################
    # Utils
    #################################################################################################################
    parser.add_argument("--run_id", type=str, default=None, help="Explicit run id (also the run directory name)")
    parser.add_argument("--run_id_note", type=str, default=None, help="Extra note inserted into the auto run id")
    parser.add_argument("--metrics_root", type=str, default="./runs", help="Root directory for run artifacts")
    parser.add_argument("--local_log_dir", type=str, default=None, help="Deprecated and ignored")
    parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases (off by default)")
    parser.add_argument("--wandb_project", type=str, default="LIBERO_simulation_test")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7, help="Random seed (for reproducibility)")
    parser.add_argument(
        "--cudaid",
        type=int,
        default=0,
        help="Deprecated and ignored; select the GPU with CUDA_VISIBLE_DEVICES",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_libero(args)
