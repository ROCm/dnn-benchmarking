#!/usr/bin/env bash
set -euo pipefail

device_target="gfx942"
tag=""
podman_args=()

usage() {
    cat <<'EOF'
Usage: ./build-dnn-benchmark-performance-image.sh [options] [-- podman-build-args...]

Build the dnn-benchmark-performance Podman image from the local workspace.

Options:
  --device-target TARGET  ROCm PyTorch GPU architecture. Default: gfx942
  --tag TAG               Podman image tag. Default: dnn-benchmark-performance:<device-target>
  -h, --help              Show this help.

Examples:
  ./build-dnn-benchmark-performance-image.sh
  ./build-dnn-benchmark-performance-image.sh --device-target gfx950
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device-target)
            [[ $# -ge 2 ]] || { echo "--device-target requires a value" >&2; exit 2; }
            device_target="$2"
            shift 2
            ;;
        --tag|-t)
            [[ $# -ge 2 ]] || { echo "--tag requires a value" >&2; exit 2; }
            tag="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            podman_args+=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v podman >/dev/null 2>&1; then
    echo "podman is required but was not found on PATH" >&2
    exit 1
fi

if [[ -z "${tag}" ]]; then
    tag="dnn-benchmark-performance:${device_target}"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"


podman build \
    -f "${script_dir}/Dockerfile.dnn-benchmark-performance-linux" \
    --build-arg "DEVICE_TARGET=${device_target}" \
    -t "${tag}" \
    "${podman_args[@]}" \
    "${repo_root}"
