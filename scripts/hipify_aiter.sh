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
echo "Done hipifying."
