#!/bin/bash
# "hipify" complete aiter code.

set -ex

PROJECT_DIR="third_party/aiter"
OUTPUT_DIR="build/hipified_aiter"
HIPIFY_CLI="third_party/hipify_torch/hipify_cli.py"

echo "Hipifying ${PROJECT_DIR} into ${OUTPUT_DIR} ..."
python3 "${HIPIFY_CLI}" \
  --project-directory "${PROJECT_DIR}" \
  --output-directory "${OUTPUT_DIR}"

echo "Post-processing hipified files to fix remaining CUDA headers..."
# (Ruturaj4): hipify isn't correctly converting some hip headers.
# Fix cuda_bf16.h include that wasn't properly converted.
find "${OUTPUT_DIR}" -name "*.hip" -exec sed -i 's/#include <cuda_bf16\.h>/#include <hip\/hip_bfloat16.h>/g' {} \;

# (Ruturaj4): Fix MHA files to use HIP version of mha_common.h.
find "${OUTPUT_DIR}" -name "*.hip" -exec sed -i 's/#include "mha_common\.h"/#include "mha_common_hip.h"/g' {} \;

echo "Done hipifying."
