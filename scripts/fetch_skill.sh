#!/bin/sh
# Download the official h3-prompt-writing skill (MiniMax-AI/MiniMax-H3, no license file → not vendored in this repo)
# at a pinned commit into src/skill/ and verify it. Used by the Dockerfile and for local tests.
set -eu
COMMIT=${MINIMAX_H3_COMMIT:-d21241f0a4b3acbb34c97dae47fa417b7065e438}
DEST=${1:-$(dirname "$0")/../src/skill}
BASE=https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/$COMMIT/skills/h3-prompt-writing
mkdir -p "$DEST"
curl -fsSL "$BASE/SKILL.md" -o "$DEST/SKILL.md"
curl -fsSL "$BASE/references/base-en.txt" -o "$DEST/base-en.txt"
cd "$DEST" && sha256sum -c - <<SUMS
a7000443588ca3f145e3b3fd8900f14e0325dc460bd811268fac89a9dc8e56d0  SKILL.md
2cfebc096a6e08370f288d468d90b60f7f9bcb938f94bf090816e910e48e75fc  base-en.txt
SUMS
