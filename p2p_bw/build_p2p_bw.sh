SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_BANDWIDTH_TEST_DIR="${SCRIPT_DIR}/../3rdparty/rocm_bandwidth_test"

mkdir -p ${ROCM_BANDWIDTH_TEST_DIR}/build
cd ${ROCM_BANDWIDTH_TEST_DIR}/build
cmake -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_MODULE_PATH="${ROCM_BANDWIDTH_TEST_DIR}/cmake_modules" -DCMAKE_PREFIX_PATH="${ROCM_BANDWIDTH_TEST_DIR}/rocr_libs" ..
make -j