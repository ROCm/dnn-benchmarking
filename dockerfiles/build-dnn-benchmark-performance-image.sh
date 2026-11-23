#!/usr/bin/env bash
set -euo pipefail

container_command="docker"
device_target="gfx942"
tag=""
rocm_version=""
rocm_libraries_ref=""
build_args=()

usage() {
    cat <<'EOF'
Usage: ./build-dnn-benchmark-performance-image.sh [options] [-- build-args...]

Build the dnn-benchmark-performance image from the local workspace.

Options:
  --container-command CMD  Container command. Default: docker
  --device-target TARGET   ROCm PyTorch GPU architecture. Default: gfx942
  --tag TAG                Image tag. Default: dnn-benchmark-performance:<device-target>
  --rocm-version VERSION   ROCm wheel version to install, for example
                           10.3.0a20261117. Default: the newest nightly that
                           Quartz reports as built for the device target,
                           selected on this host.
  --rocm-libraries-ref SHA Full 40-character rocm-libraries commit to build.
                           Default: the rocm-libraries commit pinned by the
                           selected ROCm nightly's TheRock commit (falling back
                           to the local submodule pin if Quartz has none).
  -h, --help               Show this help.

Examples:
  ./build-dnn-benchmark-performance-image.sh
  ./build-dnn-benchmark-performance-image.sh --container-command podman
  ./build-dnn-benchmark-performance-image.sh --device-target gfx950
  ./build-dnn-benchmark-performance-image.sh --rocm-version 10.3.0a20261117
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
        --rocm-version)
            [[ $# -ge 2 ]] || { echo "--rocm-version requires a value" >&2; exit 2; }
            rocm_version="$2"
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

# Select the ROCm nightly here, with the same rule as setup_env.py, and resolve
# the matching rocm-libraries commit from its TheRock commit unless the caller
# passed --rocm-libraries-ref. The image then installs and labels that matched
# build even when a newer nightly appears.
resolved="$(python3 - "${repo_root}/setup_env.py" "${device_target}" "${rocm_version}" "${rocm_libraries_ref}" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("setup_env", sys.argv[1])
setup_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_env)
device_target, rocm_version, rocm_libraries_ref = sys.argv[2:]
if rocm_version:
    build = setup_env.describe_rocm_version(rocm_version, "", "linux", device_target)
else:
    build = setup_env.select_quartz_nightly("linux", device_target)
if not rocm_libraries_ref and build.therock_commit:
    rocm_libraries_ref = setup_env.resolve_therock_rocm_libraries_commit(
        build.therock_commit
    )
print(build.rocm_version, build.index_url, rocm_libraries_ref)
PY
)"
read -r rocm_version rocm_index_url rocm_libraries_ref <<<"${resolved}"

# Fall back to the local submodule pin when Quartz had no TheRock commit (for
# example a custom --rocm-version unknown to Quartz).
if [[ -z "${rocm_libraries_ref}" ]]; then
    rocm_libraries_ref="$(git -C "${repo_root}" rev-parse HEAD:rocm-libraries 2>/dev/null || true)"
fi
if [[ -z "${rocm_libraries_ref}" ]]; then
    echo "WARNING: could not resolve a rocm-libraries commit; the image will build its moving branch." >&2
fi
echo "ROCm ${rocm_version} from ${rocm_index_url} (rocm-libraries ${rocm_libraries_ref:-moving})"

DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" "${container_command}" build \
    -f "${script_dir}/Dockerfile.dnn-benchmark-performance-linux" \
    --build-arg "DEVICE_TARGET=${device_target}" \
    --build-arg "ROCM_LIBRARIES_REF=${rocm_libraries_ref}" \
    --build-arg "ROCM_VERSION=${rocm_version}" \
    --build-arg "ROCM_TORCH_INDEX_URL=${rocm_index_url}" \
    -t "${tag}" \
    "${build_args[@]}" \
    "${repo_root}"
