#!/usr/bin/env bash
# Bump the vendored llama.cpp submodule to a new tag and re-extract the
# pre-tokenizer hash table.
#
# This script is a skeleton — Sprint 11 adds the actual submodule at
# `vendor/llama.cpp`. Sprint 06 ships the script + the shape of
# `vendor/llama_cpp_pretokenizer_hashes.json` so the compatibility
# probes (base_models/probes.py) have somewhere to read from.
#
# Usage:
#   scripts/bump-llama-cpp.sh <tag>
#       Fast-forward submodule to `<tag>`, re-extract hashes, stage.

set -euo pipefail

TAG="${1:-}"
if [ -z "$TAG" ]; then
  echo "usage: scripts/bump-llama-cpp.sh <tag>" >&2
  exit 2
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree must be clean before a submodule bump" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
VENDOR_DIR="$REPO_ROOT/vendor/llama.cpp"
HASHES_PATH="$REPO_ROOT/vendor/llama_cpp_pretokenizer_hashes.json"

if [ ! -d "$VENDOR_DIR" ]; then
  echo "error: $VENDOR_DIR missing — Sprint 11 vendors llama.cpp as a submodule" >&2
  exit 1
fi

echo "--> fetching tags in $VENDOR_DIR"
git -C "$VENDOR_DIR" fetch --tags origin

echo "--> checking out $TAG"
git -C "$VENDOR_DIR" checkout "tags/$TAG"

echo "--> re-extracting pre-tokenizer hash labels to $HASHES_PATH"
uv run python - <<'PY'
import json
import re
import sys
from pathlib import Path

repo_root = Path.cwd()
converter = repo_root / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py"
hashes_path = repo_root / "vendor" / "llama_cpp_pretokenizer_hashes.json"

source = converter.read_text(encoding="utf-8", errors="replace")
# llama.cpp declares pre-tokenizer labels inside `get_vocab_base_pre`
# via `res = "<label>"` assignments.
pattern = re.compile(r"""\bres\s*=\s*["']([^"']+)["']""")
labels = sorted(set(pattern.findall(source)))
if not labels:
    print("ERROR: no pre-tokenizer labels found in convert_hf_to_gguf.py",
          file=sys.stderr)
    sys.exit(1)

hashes_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
print(f"wrote {len(labels)} labels to {hashes_path}")
PY

echo "--> staging changes"
git -C "$REPO_ROOT" add vendor/llama.cpp vendor/llama_cpp_pretokenizer_hashes.json

cat <<EOF
Done. Review the staged diff and commit with:
  git commit -m "chore: bump llama.cpp to $TAG + refresh pre-tokenizer hashes"

Then re-run the registry probe suite:
  uv run python scripts/refresh-registry.py
EOF
