#!/usr/bin/env bash
# Pull the team's harvested activations + cleaned MCPTox pairs from origin/main
# and stage them under sae_arm/ for the SAE pipeline.
#
# Why: this branch (sae-arm-experiments) dropped diffmean/ to keep the pod
# clone slim, but the activations + pairs file we depend on still live in
# main. We use git archive / git show so the diffmean/ tree never lands in
# the working copy — only the bytes we actually need.
#
# Idempotent: skips files that are already present.
# Run order: setup_pod.sh -> fetch_team_data.sh -> run_full_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Make sure origin/main is available locally. A --single-branch clone of
# sae-arm-experiments won't have it yet; an unrestricted clone already does.
echo "[fetch] git fetch origin main ..."
git fetch origin main

# 1) Activations: diffmean/outputs/acts/  ->  sae_arm/acts/
ACTS_SENTINEL="sae_arm/acts/qwen3-v2-contrast/L20/H_pos.pt"
if [ -f "$ACTS_SENTINEL" ]; then
    echo "[fetch] sae_arm/acts/ already populated (sentinel: $ACTS_SENTINEL) — skipping"
else
    echo "[fetch] extracting diffmean/outputs/acts/ from origin/main -> sae_arm/acts/"
    mkdir -p sae_arm/acts
    # --strip-components=2 drops the "diffmean/outputs/" prefix so the tree
    # lands as sae_arm/acts/<set>/L<layer>/{H_pos,H_neg}.pt + index.jsonl.
    git archive origin/main diffmean/outputs/acts \
        | tar -x --strip-components=2 -C sae_arm/
    echo "[fetch] done: $(find sae_arm/acts -name '*.pt' | wc -l | tr -d ' ') .pt files under sae_arm/acts/"
fi

# 2) MCPTox pairs file: diffmean/outputs/mcptox_pairs.clean.jsonl  ->  sae_arm/
PAIRS="sae_arm/mcptox_pairs.clean.jsonl"
if [ -f "$PAIRS" ]; then
    echo "[fetch] $PAIRS already present — skipping"
else
    echo "[fetch] extracting mcptox_pairs.clean.jsonl from origin/main"
    git show origin/main:diffmean/outputs/mcptox_pairs.clean.jsonl > "$PAIRS"
    echo "[fetch] done: $(wc -l < "$PAIRS" | tr -d ' ') rows in $PAIRS"
fi

echo "[fetch] all team data staged under sae_arm/."
