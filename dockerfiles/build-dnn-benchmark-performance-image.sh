#!/usr/bin/env bash
set -euo pipefail

container_command="docker"
device_target="gfx942"
tag=""
build_args=()

usage() {
    cat <<'EOF'
Usage: ./build-dnn-benchmark-performance-image.sh [options] [-- build-args...]

Build the dnn-benchmark-performance image from the local workspace.

Options:
  --container-command CMD  Container command. Default: docker
  --device-target TARGET   ROCm PyTorch GPU architecture. Default: gfx942
  --tag TAG                Image tag. Default: dnn-benchmark-performance:<device-target>
  -h, --help               Show this help.

Examples:
  ./build-dnn-benchmark-performance-image.sh
  ./build-dnn-benchmark-performance-image.sh --container-command podman
  ./build-dnn-benchmark-performance-image.sh --device-target gfx950
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container-command)
            [[ $# -ge 2 ]] || { echo "--container-command requires a value" >&2; exit 2; }
            container_command="$2"
            shift 2
            ;;
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
            build_args+=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v "${container_command}" >/dev/null 2>&1; then
    echo "${container_command} is required but was not found on PATH" >&2
    exit 1
fi

if [[ -z "${tag}" ]]; then
    tag="dnn-benchmark-performance:${device_target}"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

"${container_command}" build \
    -f "${script_dir}/Dockerfile.dnn-benchmark-performance-linux" \
    --build-arg "DEVICE_TARGET=${device_target}" \
    -t "${tag}" \
    "${build_args[@]}" \
    "${repo_root}"
