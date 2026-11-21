#!/usr/bin/env bash
# Pull the OpenVLA checkpoints needed by the P0 baseline into the AutoDL data disk.
#
# Default: ONLY the baseline checkpoint (~15 GB), which is what the P0 clean/attacked
# episodes use:
#   repo  openvla/openvla-7b-finetuned-libero-spatial
#   dest  /root/autodl-tmp/checkpoints/openvla-7b-finetuned-libero-spatial
#
# The base model and the other three suite checkpoints belong to the cross-suite transfer
# stage and are gated behind --full. Enabling --full needs a data disk of >= 150 GB; the
# expanded disk is billed per day and keeps being charged while the instance is shut down.
#
# Usage:
#   bash download_weights.sh                     # baseline only -> /root/autodl-tmp/checkpoints
#   bash download_weights.sh --dest /root/autodl-tmp/ckpt
#   bash download_weights.sh --full              # all 5 repos, >= 150 GB data disk
#
# Run it in the background on day one:
#   nohup bash download_weights.sh > /root/autodl-tmp/download.log 2>&1 &

set -euo pipefail

DEST="/root/autodl-tmp/checkpoints"
MODE="baseline"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    --full)
      MODE="full"
      shift
      ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "[error] unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

export HF_ENDPOINT="https://hf-mirror.com"

# Baseline: exactly one checkpoint, ~15 GB. This is the only weight set the P0 commands use.
BASELINE_REPOS=(
  "openvla/openvla-7b-finetuned-libero-spatial"
)

# 跨套件迁移阶段再启用。以下 4 个仓库合计约 60 GB，加上 baseline 共约 75 GB。
EXTRA_REPOS=(
  "openvla/openvla-7b"
  "openvla/openvla-7b-finetuned-libero-object"
  "openvla/openvla-7b-finetuned-libero-goal"
  "openvla/openvla-7b-finetuned-libero-10"
)

REPOS=("${BASELINE_REPOS[@]}")
if [ "${MODE}" = "full" ]; then
  REPOS+=("${EXTRA_REPOS[@]}")
fi

mkdir -p "${DEST}"

echo "[info] mode=${MODE}"
echo "[info] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[info] destination=${DEST}"

if [ "${MODE}" = "baseline" ]; then
  echo "[info] baseline checkpoint: openvla/openvla-7b-finetuned-libero-spatial (~15 GB)"
  echo "[info]   -> ${DEST}/openvla-7b-finetuned-libero-spatial"
  echo "[info] base model and the other 3 suites are NOT downloaded; pass --full to enable them"
else
  echo "[warn] --full downloads ~75 GB in total; the AutoDL default data disk is 50 GB."
  echo "[warn] Expand the data disk to >= 150 GB first; the expansion is billed per day and"
  echo "[warn] keeps being billed while the instance is shut down."
fi

# Free-space guard: df -Pk reports 1K blocks, so $4/1024/1024 is GB.
avail_gb=$(df -Pk "${DEST}" | awk 'NR==2 {print int($4 / 1024 / 1024)}')
need_gb=20
if [ "${MODE}" = "full" ]; then
  need_gb=80
fi
if [ "${avail_gb}" -lt "${need_gb}" ]; then
  echo "[error] free space ${avail_gb} GB < required ${need_gb} GB under ${DEST}" >&2
  echo "[error] expand the data disk (AutoDL console -> 扩容), or pass --dest to a larger disk" >&2
  exit 1
fi
echo "[info] free space ${avail_gb} GB (required ${need_gb} GB)"

for repo in "${REPOS[@]}"; do
  name="${repo##*/}"
  if [ -f "${DEST}/${name}/model.safetensors.index.json" ]; then
    echo "[skip] ${name} already complete"
    continue
  fi
  echo "[download] ${repo} -> ${DEST}/${name}"
  huggingface-cli download "${repo}" --local-dir "${DEST}/${name}"
done

echo "[done] checkpoints under ${DEST}"
du -sh "${DEST}"/* 2>/dev/null || true
