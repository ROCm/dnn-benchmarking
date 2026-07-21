#!/usr/bin/env bash
set -euo pipefail

device_target="gfx942"
gpu_target=""
tag=""
docker_args=()

usage() {
    cat <<'EOF'
Usage: ./build-dnn-benchmark-performance-image.sh [options] [-- docker-build-args...]

Build the dnn-benchmark-performance Docker image from the local workspace.

Options:
  --device-target TARGET  ROCm PyTorch GPU architecture. Default: gfx942
  --gpu-target TARGET     Optional hipDNN/provider CMake GPU_TARGETS override. Default: <device-target>
  --tag TAG               Docker image tag. Default: dnn-benchmark-performance:<device-target>
  -h, --help              Show this help.

Examples:
  ./build-dnn-benchmark-performance-image.sh
  ./build-dnn-benchmark-performance-image.sh --device-target gfx942
  ./build-dnn-benchmark-performance-image.sh --device-target gfx950 --gpu-target gfx950 -- --no-cache --progress=plain
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device-target)
            [[ $# -ge 2 ]] || { echo "--device-target requires a value" >&2; exit 2; }
            device_target="$2"
            shift 2
            ;;
        --gpu-target)
            [[ $# -ge 2 ]] || { echo "--gpu-target requires a value" >&2; exit 2; }
            gpu_target="$2"
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
            docker_args+=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required but was not found on PATH" >&2
    exit 1
fi

if [[ -z "${tag}" ]]; then
    tag="dnn-benchmark-performance:${device_target}"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"


DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" docker build \
    -f "${script_dir}/Dockerfile.dnn-benchmark-performance-linux" \
    --build-arg "DEVICE_TARGET=${device_target}" \
    --build-arg "GPU_TARGET=${gpu_target}" \
    -t "${tag}" \
    "${docker_args[@]}" \
    "${repo_root}"
