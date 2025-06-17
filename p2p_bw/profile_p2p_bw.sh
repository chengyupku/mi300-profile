SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_BANDWIDTH_TEST_DIR="${SCRIPT_DIR}/../3rdparty/rocm_bandwidth_test"

echo "Printing system topology and allocatable memory info..."
${ROCM_BANDWIDTH_TEST_DIR}/build/rocm-bandwidth-test -t

echo "Perform Unidirectional Copy involving all device combinations..."
${ROCM_BANDWIDTH_TEST_DIR}/build/rocm-bandwidth-test -a

echo "Perform Bidirectional Copy involving all device combinations..."
${ROCM_BANDWIDTH_TEST_DIR}/build/rocm-bandwidth-test -A