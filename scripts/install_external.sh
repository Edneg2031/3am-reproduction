#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
mkdir -p "${EXTERNAL_DIR}"

if [ ! -d "${EXTERNAL_DIR}/sam2/.git" ]; then
  git clone https://github.com/facebookresearch/sam2.git "${EXTERNAL_DIR}/sam2"
fi
python -m pip install -e "${EXTERNAL_DIR}/sam2"

if [ ! -d "${EXTERNAL_DIR}/must3r/.git" ]; then
  git clone https://github.com/naver/must3r.git "${EXTERNAL_DIR}/must3r"
fi
python -m pip install -e "${EXTERNAL_DIR}/must3r" || true

echo "External repositories are installed under ${EXTERNAL_DIR}."
