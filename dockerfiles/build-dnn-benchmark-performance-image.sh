#!/usr/bin/env bash
set -euo pipefail

container_command="docker"
device_target="gfx942"
tag=""
rocm_libraries_ref="develop"
build_args=()

usage() {
    cat <<'EOF'
Usage: ./build-dnn-benchmark-performance-image.sh [options] [-- build-args...]

Build the dnn-benchmark-performance image from the local workspace.

Options:
  --container-command CMD  Container command. Default: docker
  --device-target TARGET   ROCm PyTorch GPU architecture. Default: gfx942
  --tag TAG                Image tag. Default: dnn-benchmark-performance:<device-target>
  --rocm-libraries-ref REF rocm-libraries to build: a full 40-character commit,
                           or a branch or tag name, resolved to its commit on
                           this host. Default: develop
  -h, --help               Show this help.

Examples:
  ./build-dnn-benchmark-performance-image.sh
  ./build-dnn-benchmark-performance-image.sh --container-command podman
  ./build-dnn-benchmark-performance-image.sh --device-target gfx950
  ./build-dnn-benchmark-performance-image.sh --rocm-libraries-ref <commit>
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
        --rocm-libraries-ref)
            [[ $# -ge 2 ]] || { echo "--rocm-libraries-ref requires a value" >&2; exit 2; }
            rocm_libraries_ref="$2"
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

# Resolve the ref to one commit here, with the same rule as setup_env.py, so
# the image builds and labels that commit even when the branch moves.
rocm_libraries_commit="$(python3 - "${repo_root}/setup_env.py" "${rocm_libraries_ref}" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("setup_env", sys.argv[1])
setup_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_env)
print(setup_env.resolve_remote_commit(setup_env.ROCM_LIBRARIES_URL, sys.argv[2]))
PY
)"
echo "rocm-libraries ${rocm_libraries_ref}: ${rocm_libraries_commit}"

DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" "${container_command}" build \
    -f "${script_dir}/Dockerfile.dnn-benchmark-performance-linux" \
    --build-arg "DEVICE_TARGET=${device_target}" \
    --build-arg "ROCM_LIBRARIES_COMMIT=${rocm_libraries_commit}" \
    -t "${tag}" \
    "${build_args[@]}" \
    "${repo_root}"
