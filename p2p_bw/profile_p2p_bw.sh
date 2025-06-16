SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_BANDWIDTH_TEST_DIR="${SCRIPT_DIR}/../3rdparty/rocm_bandwidth_test"

${ROCM_BANDWIDTH_TEST_DIR}/build/rocm-bandwidth-test -A