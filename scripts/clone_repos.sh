#!/bin/bash
# TSG SOTA Benchmark — Clone Method Repositories
#
# Clones all 8 method repositories into repos/.
# Run from the project root: bash scripts/clone_repos.sh
#
# If a repo already exists, it is skipped (use "git pull" to update).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPOS_DIR="$(dirname "$SCRIPT_DIR")/repos"

mkdir -p "$REPOS_DIR"
cd "$REPOS_DIR"

REPOS=(
    "https://github.com/jsyoon0823/TimeGAN.git"
    "https://github.com/ratschlab/RGAN.git"
    "https://github.com/Jinsung-Jeon/GT-GAN.git"
    "https://github.com/abudesai/timeVAE.git"
    "https://github.com/ahmedmalaa/Fourier-flows.git"
    "https://github.com/ermongroup/CSDI.git"
    "https://github.com/amazon-science/unconditional-time-series-diffusion.git"
    "https://github.com/Y-debug-sys/Diffusion-TS.git"
)

echo "Cloning 8 method repositories into $REPOS_DIR ..."
echo ""

for REPO_URL in "${REPOS[@]}"; do
    REPO_NAME=$(basename "$REPO_URL" .git)
    if [ -d "$REPO_NAME" ]; then
        echo "  [SKIP] $REPO_NAME already exists"
    else
        echo "  [CLONE] $REPO_NAME ..."
        git clone "$REPO_URL" "$REPO_NAME" 2>&1 | sed 's/^/         /'
        echo "  [DONE]  $REPO_NAME"
    fi
done

echo ""
echo "All repositories cloned. Summary:"
echo "--------------------------------"
for REPO_URL in "${REPOS[@]}"; do
    REPO_NAME=$(basename "$REPO_URL" .git)
    if [ -d "$REPO_NAME" ]; then
        COMMITS=$(cd "$REPO_NAME" && git rev-list --count HEAD 2>/dev/null || echo "?")
        echo "  $REPO_NAME ($COMMITS commits)"
    else
        echo "  $REPO_NAME (MISSING)"
    fi
done
