from typing import Optional
import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl
from _aiter_triton._triton_kernels.gemm_a8w8 import _gemm_a8w8_kernel, _get_config 
from _aiter_triton.utils.device_info import get_num_xcds


def gemm_a8w8(
    x:jnp.ndarray,
    w:jnp.ndarray,
    x_scale:jnp.ndarray,
    w_scale:jnp.ndarray,
    bias: Optional[jnp.ndarray]=None,
    dtype: Optional[float] = jnp.bfloat16,
    config: Optional[jnp.ndarray] = None
):
    """
    Computes the 8 bit matmul Y = X x WT, applies a conversion scale and optionally adds a bias
    to the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - X_scale: First scale tensor with shape (M, 1).
    - W_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).

    Returns:
    - Y: The output matrix with shape (M, N).
    """
    M, K = x.shape
    N, K = w.shape
    w = w.T

    x_strides = jt.strides_from_shape(x.shape)
    w_strides = jt.strides_from_shape(w.shape)
    y_strides = jt.strides_from_shape((M, N))

    if config is None:
        config = _get_config(M, N, K)

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    out_shape = jax.ShapeDtypeStruct(shape=(M, N), dtype=x.dtype)
    return jt.triton_call(
        x,
        w,
        x_scale,
        w_scale,
        bias,
        kernel=_gemm_a8w8_kernel,
        out_shape=out_shape,
        grid=grid,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        M=M,
        N=N,
        K=K,
        stride_am=x_strides[0],
        stride_ak=x_strides[1],
        stride_bk=w_strides[0],
        stride_bn=w_strides[1],
        stride_cm=y_strides[0],
        stride_cn=y_strides[1],
        HAS_BIAS=True,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        NUM_XCDS=get_num_xcds(),
    )

    return y

def main():
    M, N, K = 64, 32, 128
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(k1, (M, K), dtype=jnp.float32)
    w = jax.random.normal(k2, (N, K), dtype=jnp.float32)

    max_x = jnp.amax(jnp.abs(x), axis=1, keepdims=True)
    x_scale = max_x / 240.0
    x = x / x_scale
    x = x.astype(jnp.float8_e4m3fnuz)

    max_weight = jnp.amax(jnp.abs(w),axis=0, keepdims=True)
    w_scale = max_weight / 240.0
    w = w / w_scale
    w = w.astype(jnp.float8_e4m3fnuz)

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    bias = jax.random.normal(k1, (1, N), dtype=jnp.float32) 

    jax_out = gemm_a8w8(x, w, x_scale, w_scale, bias).block_until_ready()



if __name__ == "__main__":
    main()
