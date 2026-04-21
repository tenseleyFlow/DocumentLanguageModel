#!/usr/bin/env bash
# Bump the vendored llama.cpp submodule, build its tools, and refresh
# the pre-tokenizer hash table.
#
# Usage:
#   scripts/bump-llama-cpp.sh bump <tag>
#       Fast-forward submodule to <tag>, re-extract hashes, write VERSION,
#       stage changes.
#   scripts/bump-llama-cpp.sh build
#       Build `llama-quantize` (+ siblings) via cmake. Idempotent.
#   scripts/bump-llama-cpp.sh refresh-labels
#       Regenerate vendor/llama_cpp_pretokenizer_hashes.json from the
#       current submodule contents. Does not touch the submodule itself.
#   scripts/bump-llama-cpp.sh probe-vl-arch
#       Re-run the VL arch probe (Sprint 35.4) and cache the verdicts
#       under vendor/llama_cpp_vl_arch_support.json. Fast path for the
#       runtime probe; omitted or stale means the runtime falls back to
#       a live scan.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
VENDOR_DIR="$REPO_ROOT/vendor/llama.cpp"
HASHES_PATH="$REPO_ROOT/vendor/llama_cpp_pretokenizer_hashes.json"
VL_ARCH_PATH="$REPO_ROOT/vendor/llama_cpp_vl_arch_support.json"
VERSION_PATH="$VENDOR_DIR/VERSION"

cmd="${1:-}"

refresh_labels() {
  echo "--> re-extracting pre-tokenizer hash labels to $HASHES_PATH"
  uv run python - <<'PY'
import json
import re
import sys
from pathlib import Path

repo_root = Path.cwd()
converter = repo_root / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py"
hashes_path = repo_root / "vendor" / "llama_cpp_pretokenizer_hashes.json"

if not converter.is_file():
    print(f"ERROR: {converter} not found", file=sys.stderr)
    sys.exit(1)

source = converter.read_text(encoding="utf-8", errors="replace")
pattern = re.compile(r"""\bres\s*=\s*["']([^"']+)["']""")
labels = sorted(set(pattern.findall(source)))
if not labels:
    print("ERROR: no pre-tokenizer labels found in convert_hf_to_gguf.py",
          file=sys.stderr)
    sys.exit(1)

hashes_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
print(f"wrote {len(labels)} labels to {hashes_path}")
PY
}

probe_vl_arch() {
  echo "--> probing VL arch support in vendored llama.cpp"
  uv run python - <<'PY'
import json
import sys
from pathlib import Path

# Import dlm's probe directly — this script runs from the repo root
# with dlm importable via `uv run`. Failing imports abort with a
# readable error; no silent half-cache files.
sys.path.insert(0, str(Path.cwd() / "src"))
from dlm.base_models import BASE_MODELS
from dlm.export.arch_probe import SupportLevel, clear_cache, probe_gguf_arch

# Fresh probe — the cache may carry a stale verdict from an earlier
# run in the same process; clear before enumerating.
clear_cache()

out_path = Path.cwd() / "vendor" / "llama_cpp_vl_arch_support.json"
entries: dict[str, dict[str, str | None]] = {}
for key, spec in BASE_MODELS.items():
    if spec.modality != "vision-language":
        continue
    result = probe_gguf_arch(spec.architecture)
    entries[key] = {
        "architecture": spec.architecture,
        "support": result.support.value,
        "llama_cpp_tag": result.llama_cpp_tag,
        "reason": result.reason,
    }

payload = {
    "schema": 1,
    "bases": entries,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

# Print a summary table for operators.
print(f"wrote {len(entries)} VL verdicts to {out_path}")
for key, entry in sorted(entries.items()):
    print(f"  {entry['support']:<12} {key}  ({entry['architecture']})")
PY
}

do_bump() {
  local tag="${1:-}"
  if [ -z "$tag" ]; then
    echo "usage: scripts/bump-llama-cpp.sh bump <tag>" >&2
    exit 2
  fi
  if [ -n "$(git status --porcelain)" ]; then
    echo "error: working tree must be clean before a submodule bump" >&2
    exit 1
  fi
  if [ ! -d "$VENDOR_DIR" ]; then
    echo "error: $VENDOR_DIR missing — initialize the submodule first:" >&2
    echo "  git submodule add https://github.com/ggerganov/llama.cpp vendor/llama.cpp" >&2
    exit 1
  fi

  echo "--> fetching tags in $VENDOR_DIR"
  git -C "$VENDOR_DIR" fetch --tags origin
  echo "--> checking out $tag"
  git -C "$VENDOR_DIR" checkout "tags/$tag"

  echo "--> writing $VERSION_PATH"
  echo "$tag" > "$VERSION_PATH"

  refresh_labels
  probe_vl_arch

  echo "--> staging changes"
  git -C "$REPO_ROOT" add \
    vendor/llama.cpp \
    vendor/llama_cpp_pretokenizer_hashes.json \
    vendor/llama_cpp_vl_arch_support.json

  cat <<EOF
Done. Review the staged diff and commit with:
  git commit -m "chore: bump llama.cpp to $tag + refresh pre-tokenizer hashes"

Then build the binaries:
  scripts/bump-llama-cpp.sh build

And re-run the registry probe suite:
  uv run python scripts/refresh-registry.py
EOF
}

do_build() {
  if [ ! -d "$VENDOR_DIR" ]; then
    echo "error: $VENDOR_DIR missing — run 'bump <tag>' first" >&2
    exit 1
  fi
  echo "--> configuring llama.cpp via cmake"
  cmake -S "$VENDOR_DIR" -B "$VENDOR_DIR/build" -DCMAKE_BUILD_TYPE=Release
  # `llama-quantize` does the actual per-tensor quantization; `llama-imatrix`
  # produces the importance-matrix file we feed to quantize for k-quant
  # calibration (Sprint 11.6). Both are required for the full export
  # pipeline; building them separately means a missing target fails the
  # build loudly rather than silently shipping a half-built toolchain.
  for target in llama-quantize llama-imatrix; do
    echo "--> building $target"
    cmake --build "$VENDOR_DIR/build" --target "$target" --config Release
    if [ ! -f "$VENDOR_DIR/build/bin/$target" ]; then
      echo "error: build finished but $target not found under build/bin" >&2
      exit 1
    fi
    echo "OK: $VENDOR_DIR/build/bin/$target"
  done
}

case "$cmd" in
  bump)
    do_bump "${2:-}"
    ;;
  build)
    do_build
    ;;
  refresh-labels)
    refresh_labels
    ;;
  probe-vl-arch)
    probe_vl_arch
    ;;
  "")
    echo "usage: scripts/bump-llama-cpp.sh <bump|build|refresh-labels|probe-vl-arch> [args]" >&2
    exit 2
    ;;
  *)
    echo "unknown command: $cmd" >&2
    echo "usage: scripts/bump-llama-cpp.sh <bump|build|refresh-labels|probe-vl-arch> [args]" >&2
    exit 2
    ;;
esac
