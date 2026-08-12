#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT=${1:?usage: install_prebuilt_corex.sh VLLM_ROOT}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR=${SCRIPT_DIR}/prebuilt/corex-3.2.3-ivcore10
MANIFEST=${BUNDLE_DIR}/SHA256SUMS

[[ -d "$VLLM_ROOT" ]] || {
    printf 'vLLM root does not exist: %s\n' "$VLLM_ROOT" >&2
    exit 2
}
[[ -f "$MANIFEST" ]] || {
    printf 'prebuilt CoreX manifest is missing: %s\n' "$MANIFEST" >&2
    exit 2
}

mapfile -t artifacts < <(awk '{print $2}' "$MANIFEST")
[[ "${#artifacts[@]}" -eq 13 ]] || {
    printf 'expected 13 prebuilt CoreX artifacts, found %s\n' \
        "${#artifacts[@]}" >&2
    exit 2
}

for artifact in "${artifacts[@]}"; do
    [[ "$artifact" == corex_*.so && "$artifact" != */* ]] || {
        printf 'invalid prebuilt artifact name: %s\n' "$artifact" >&2
        exit 2
    }
done

(
    cd "$BUNDLE_DIR"
    sha256sum --strict --check SHA256SUMS
)

for artifact in "${artifacts[@]}"; do
    install -m 0755 "$BUNDLE_DIR/$artifact" "$VLLM_ROOT/$artifact"
done

python3 - "$VLLM_ROOT" "${artifacts[@]}" <<'PY'
import pathlib
import struct
import sys

root = pathlib.Path(sys.argv[1])
for name in sys.argv[2:]:
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"installed CoreX extension is empty: {path}")
    header = path.read_bytes()[:20]
    if len(header) < 20 or header[:4] != b"\x7fELF":
        raise SystemExit(f"installed CoreX extension is not ELF: {path}")
    if header[4:6] != b"\x02\x01":
        raise SystemExit(
            f"installed CoreX extension is not 64-bit little-endian ELF: {path}")
    machine = struct.unpack_from("<H", header, 18)[0]
    if machine != 62:
        raise SystemExit(
            f"installed CoreX extension is not x86-64 ELF: {path} machine={machine}")
    print(f"[ok] installed prebuilt CoreX extension {path}")
PY
