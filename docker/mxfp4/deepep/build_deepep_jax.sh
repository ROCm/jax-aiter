#!/usr/bin/env bash
# Builds Primus-Turbo's intranode DeepEP JAX FFI module alone (no rocSHMEM,
# no composable_kernel GEMMs) and assembles an importable `primus_turbo`
# overlay under $OUT/pkg. Primus-Turbo sources are read, never modified.
# Run inside the container as root; add $OUT/pkg to PYTHONPATH.
#   PT=<Primus-Turbo checkout> [PT_COMMIT=<sha>] [OUT=<dir>] bash build_deepep_jax.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PT=$(realpath "${PT:?set PT to a Primus-Turbo checkout}")
OUT=${OUT:-$HERE/build}
ARCH=${ARCH:-gfx950}
HIPCC=/usr/local/bin/hipcc

PY=python
EXT_SUFFIX=$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
FFI_INC=$($PY -c 'import jax.ffi; print(jax.ffi.include_dir())')
PYBIND_INC=$($PY -c 'import pybind11; print(pybind11.get_include())')
PY_INC=$($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
HIPBLASLT_INC=$(dirname "$(dirname "$($PY -c 'import glob; print(glob.glob("/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/include/hipblaslt/hipblaslt.h")[0])')")")

mkdir -p "$OUT/obj" "$OUT/pkg"
# PT_COMMIT names the sources when $PT is not a Git checkout (an image build).
echo "PT=$PT commit=${PT_COMMIT:-$(git -c safe.directory="$PT" -C "$PT" rev-parse --short HEAD)} ARCH=$ARCH"
echo "hipcc: $($HIPCC --version | head -1)"

COMMON=(-O3 -std=c++20 -fPIC -fgpu-rdc --offload-arch="$ARCH" -DPRIMUS_TURBO_${ARCH^^}
        -DDISABLE_ROCSHMEM -Wno-unknown-warning-option
        -I"$PT/csrc" -I"$PT/csrc/include" -I"$FFI_INC" -I"$PYBIND_INC" -I"$PY_INC" -I"$HIPBLASLT_INC")
DEVICE=(-DHIP_ENABLE_WARP_SYNC_BUILTINS=1
        -U__HIP_NO_HALF_OPERATORS__ -U__HIP_NO_HALF_CONVERSIONS__
        -U__HIP_NO_BFLOAT16_OPERATORS__ -U__HIP_NO_BFLOAT16_CONVERSIONS__
        -U__HIP_NO_BFLOAT162_OPERATORS__ -U__HIP_NO_BFLOAT162_CONVERSIONS__
        -fno-offload-uniform-block -include "$HERE/launch_compat.h"
        -mllvm --lsr-drop-solution=1 -mllvm -enable-post-misched=0 -mllvm -amdgpu-early-inline-all=true)

SOURCES=(
  "$PT/csrc/kernels/deep_ep/intranode.cu"
  "$PT/csrc/kernels/deep_ep/layout.cu"
  "$PT/csrc/kernels/deep_ep/runtime.cu"
  "$PT/csrc/kernels/deep_ep/internode.cu"
  "$PT/csrc/jax/deep_ep/deep_ep.cpp"
  "$PT/csrc/jax/deep_ep/handler.cpp"
  "$HERE/bindings_deepep_jax.cpp"
)

OBJS=()
for src in "${SOURCES[@]}"; do
  obj="$OUT/obj/$(basename "$src").o"
  OBJS+=("$obj")
  echo "compile $(basename "$src")"
  if [[ $src == *.cu ]]; then
    $HIPCC "${COMMON[@]}" "${DEVICE[@]}" -x hip -c "$src" -o "$obj" &
  else
    $HIPCC "${COMMON[@]}" -fvisibility=hidden -x hip -c "$src" -o "$obj" &
  fi
done
for job in $(jobs -p); do wait "$job"; done

SO="$OUT/pkg/primus_turbo/jax/_C$EXT_SUFFIX"
mkdir -p "$(dirname "$SO")"
echo "link $SO"
$HIPCC -shared -fgpu-rdc --hip-link --offload-arch="$ARCH" "${OBJS[@]}" -o "$SO" -lhipblaslt
# Python symbols resolve at import; any other undefined symbol is a missing source.
if nm -D --undefined-only "$SO" | c++filt | grep -E 'primus_turbo|deep_ep'; then
  echo "unresolved Primus-Turbo symbols" >&2
  exit 1
fi

P="$OUT/pkg/primus_turbo"
J="$P/jax"
mkdir -p "$J/lax" "$J/primitive"
ln -sfn "$PT/primus_turbo/__init__.py" "$P/__init__.py"
ln -sfn "$PT/primus_turbo/jax/__init__.py" "$J/__init__.py"
ln -sfn "$PT/primus_turbo/jax/deep_ep" "$J/deep_ep"
: > "$J/lax/__init__.py"
# Zero-size `jnp.array([])` constants hoisted into a shard_map segfault Shardy
# canonicalization in this XLA build; `jnp.zeros((0,))` lowers to a broadcast.
mkdir -p "$J/lax/moe"
for f in __init__.py moe_utils.py; do ln -sfn "$PT/primus_turbo/jax/lax/moe/$f" "$J/lax/moe/$f"; done
sed -E 's/jnp\.array\(\[\], dtype=([^)]+)\)/jnp.zeros((0,), dtype=\1)/' \
  "$PT/primus_turbo/jax/lax/moe/moe_dispatch_combine.py" > "$J/lax/moe/moe_dispatch_combine.py"
if grep -n 'jnp\.array(\[\]' "$J/lax/moe/moe_dispatch_combine.py"; then
  echo "empty-constant rewrite incomplete" >&2
  exit 1
fi
cp "$HERE/primitive__init__.py" "$J/primitive/__init__.py"
ln -sfn "$PT/primus_turbo/jax/primitive/moe" "$J/primitive/moe"
ln -sfn "$PT/primus_turbo/jax/primitive/_batching.py" "$J/primitive/_batching.py"

PYTHONPATH="$OUT/pkg${PYTHONPATH:+:$PYTHONPATH}" $PY - <<'EOF'
import primus_turbo.jax as ptj
ptj.initialize()
from primus_turbo.jax._C import registrations
print("registered:", sorted(registrations()))
from primus_turbo.jax.lax.moe import setup, moe_dispatch, moe_combine
print("import OK")
EOF
