#!/usr/bin/env bash
# Build a Codabench ZIP for a variant under variants/<name>/.
#
# The variant directory contains an override for model.py (and optionally
# labeling.py). submit.py reads files from REPO_ROOT, so we briefly swap
# the variant's overrides into root, build, and restore. The cleanup trap
# uses `mv -f` because the build re-creates the destination and `mv -n`
# would silently refuse to restore (see repository memory dated 2026-05-20).
#
# Usage:
#   ./build_variant.sh const_0p5
set -euo pipefail
VARIANT="${1:?usage: $0 <variant_name>}"

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

VARIANT_DIR="variants/$VARIANT"
MANIFEST="manifests/$VARIANT.json"

[[ -f "$VARIANT_DIR/model.py" ]] || { echo "missing: $VARIANT_DIR/model.py" >&2; exit 1; }
[[ -f "$MANIFEST"             ]] || { echo "missing: $MANIFEST" >&2; exit 1; }

# Save current root model.py / labeling.py so we can restore them after build.
mv -f model.py model.py.user
HAS_LABEL_OVERRIDE=0
if [[ -f "$VARIANT_DIR/labeling.py" ]]; then
    mv -f labeling.py labeling.py.user
    HAS_LABEL_OVERRIDE=1
fi

restore() {
    mv -f model.py.user model.py
    if [[ "$HAS_LABEL_OVERRIDE" = "1" ]]; then
        mv -f labeling.py.user labeling.py
    fi
}
trap restore EXIT

cp "$VARIANT_DIR/model.py" model.py
if [[ "$HAS_LABEL_OVERRIDE" = "1" ]]; then
    cp "$VARIANT_DIR/labeling.py" labeling.py
fi

python submit.py build "$MANIFEST"
