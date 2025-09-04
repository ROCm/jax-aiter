from typing import Optional
import jax
import jax.numpy as jnp
import jax_triton as jt
import numpy as np 
from torch2jax import j2t
import triton
import triton.language as tl
from _aiter_triton._triton_kernels.gemm_a16w16 import _gemm_a16_w16_kernel, _get_config
from _aiter_triton._triton_kernels.activation import _get_activation_from_str


def gemm_a16w16(
    x,
    w,
    dtype: Optional[float] = jnp.bfloat16,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
):
    M, K = x.shape
    N, K = w.shape
    w = w.T
  
    x_strides = jt.strides_from_shape(x.shape)
    w_strides = jt.strides_from_shape(w.shape)
    y_strides = jt.strides_from_shape((M, N))

    if config is None:
        config = _get_config(M, N, K)

    grid =  (  
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    out_shape = jax.ShapeDtypeStruct(shape=(M, N), dtype=dtype)
    return jt.triton_call(
        x,
        w,
        kernel=_gemm_a16_w16_kernel,
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
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        cache_modifier=config["cache_modifier"],
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        )
    


#NOTE: this is just for demostration only. This code should go in a pytest unit test. 
#The API wrappers for torch in AITER don't have main function. 
#https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/gemm_a16w16.py
#
#UNIT Test case example
#https://github.com/ROCm/aiter/blob/main/op_tests/triton_tests/test_gemm_a16w16.py
def main():
    M, N, K = 64, 32, 128
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    
    #Note the ideal layout for Triton GEMMs is TN ie Matrix A is row-major and Matrix B is column-major. 
    #However, Pytorch and JAX frameworks support row-major layout. We play a trick here by
    #declaring matrix B to be of shape N,K and then transpose. Transposing doesn't change the memory layout. Just
    #the strides and shape of the nd-array object.
    x = jax.random.normal(k1, (M, K), dtype=jnp.float16)
    w = jax.random.normal(k2, (N, K), dtype=jnp.float16) 

    jax_triton_out = gemm_a16w16(x, w, dtype=jnp.float16, activation="relu").block_until_ready()

    #Verify using JAX reference. See if conversion to numpy can be avoided since numpy doesn't support
    #some important DL/ML workload dtypes like bfloat16, etc.
    jax_out = jax.nn.relu(jnp.linalg.matmul(x, w.T))

    #Here we are using triton API to compare. https://triton-lang.org/main/python-api/generated/triton.testing.assert_close.html
    #This requires converting to numpy because API only accepts numpy or torch tensor. Again, this is not ideal
    #because of lacking datatype support in Numpy. May be Jax array can be converted to torch tensors before
    #comparison.
    #Can also use torch.testing. See https://github.com/ROCm/aiter/blob/main/op_tests/triton_tests/test_gemm_a16w16.py#L131
    triton.testing.assert_close(np.asarray(jax_out), np.asarray(jax_triton_out), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    main()


