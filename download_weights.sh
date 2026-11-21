#!/usr/bin/env bash
# Pull the OpenVLA checkpoints used by the P0 baseline into the AutoDL data disk.
#
# Weights are ~15 GB each (~76 GB total for the 5 repos below). They must never enter git:
# they live outside the repo, on the persistent data disk.
#
# Usage:
#   bash download_weights.sh                      # -> /root/autodl-tmp/checkpoints
#   bash download_weights.sh /root/autodl-tmp/ckpt
#
# Run this in the background on day one, e.g.:
#   nohup bash download_weights.sh > /root/autodl-tmp/download.log 2>&1 &

set -euo pipefail

DEST="${1:-/root/autodl-tmp/checkpoints}"
export HF_ENDPOINT="https://hf-mirror.com"

REPOS=(
  "openvla/openvla-7b"
  "openvla/openvla-7b-finetuned-libero-spatial"
  "openvla/openvla-7b-finetuned-libero-object"
  "openvla/openvla-7b-finetuned-libero-goal"
  "openvla/openvla-7b-finetuned-libero-10"
)

mkdir -p "${DEST}"
echo "[info] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[info] destination=${DEST}"

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
