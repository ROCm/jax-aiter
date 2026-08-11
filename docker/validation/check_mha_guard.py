"""The default wheel must explain how to install optional MHA libraries."""

try:
    import jax_aiter.mha  # noqa: F401
except ModuleNotFoundError as exc:
    message = str(exc)
    assert "jax-aiter-fetch-mha" in message, message
    print("MHA optional-library guard PASS")
else:
    raise AssertionError("default wheel imported MHA before optional JIT fetch")
