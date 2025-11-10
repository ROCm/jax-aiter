import time
import copy
import os
import jax
import jax.numpy as jnp

try:
    # TODO: (Ruturaj4) See if we want to implement this.
    from jax_aiter import logger
except Exception:  # fallback
    import logging

    logger = logging.getLogger("jax_aiter_test")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)


def _run_iters(num_iters, func, *args, **kwargs):
    out = None
    for _ in range(num_iters):
        out = func(*args, **kwargs)
    return out


def _run_iters_rotate(num_iters, func, rotate_args):
    out = None
    n = len(rotate_args)
    for i in range(num_iters):
        args, kwargs = rotate_args[i % n]
        out = func(*args, **kwargs)
    return out


def benchmark():
    """
    JAX version of aiter.test_common.benchmark

    Usage:
        @benchmark()
        def foo(x):
            return {"res": x + 1}
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # we don't have torch-style tensor introspection here,
            # just call the function and return its dict (if any)
            ret = func(*args, **kwargs)
            if isinstance(ret, dict):
                return ret
            return {"ret": ret}

        return wrapper

    return decorator


def run_perftest(
    func,
    *args,
    num_iters=101,
    num_warmup=2,
    num_rotate_args=0,
    **kwargs,
):
    """
    JAX version of aiter.test_common.run_perftest

    - warmup `num_warmup` times
    - then run `num_iters` times
    - sync after each run to get real device time
    - return (last_output, avg_us_per_iter)
    """
    # figure out rotation args (like torch version)
    if num_rotate_args and num_rotate_args > 0:
        rotate_args = [
            (copy.deepcopy(args), copy.deepcopy(kwargs))
            for _ in range(num_rotate_args - 1)
        ] + [(args, kwargs)]
    else:
        rotate_args = [(args, kwargs)]

    # warmup
    for _ in range(num_warmup):
        out = func(*args, **kwargs)
        # make sure we materialize device work
        out = jax.tree.map(lambda x: x, out)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            out,
        )

    # real runs
    lat_us = []
    out = None
    for i in range(num_iters):
        call_args, call_kwargs = rotate_args[i % len(rotate_args)]
        t0 = time.perf_counter()
        out = func(*call_args, **call_kwargs)
        # sync
        out = jax.tree.map(lambda x: x, out)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            out,
        )
        t1 = time.perf_counter()
        lat_us.append((t1 - t0) * 1e6)

    avg_us = sum(lat_us) / len(lat_us)
    logger.info(f"[JAX perftest] avg: {avg_us:.2f} us/iter over {num_iters} iters")

    return out, avg_us
