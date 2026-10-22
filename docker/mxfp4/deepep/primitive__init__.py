"""Primus-Turbo `primus_turbo/jax/primitive/__init__.py`, registering only the DeepEP MoE primitives."""

from typing import Any, Dict

from jax.extend.core import Primitive

IMPL_TABLE: Dict[Primitive, Any] = {}
ABSTRACT_EVAL_TABLE: Dict[Primitive, Any] = {}
LOWERING_TABLE: Dict[Primitive, Any] = {}

TRANSPOSE_TABLE: Dict[Primitive, Any] = {}
BATCHING_TABLE: Dict[Primitive, Any] = {}

from . import moe  # noqa: E402,F401
