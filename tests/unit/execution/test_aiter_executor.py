# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""AITER executor: the JIT directory and toolchain set before ``import aiter``,
and the arguments of each AITER entry point."""

import importlib.machinery
import importlib.util
import os
import sys
import types

import pytest

from dnn_benchmarking.execution import aiter_executor
from dnn_benchmarking.execution.aiter_sdpa import AiterBackend
from dnn_benchmarking.execution.sdpa_graph import DenseSdpa


def _executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def venv(tmp_path, monkeypatch):
    """A venv interpreter whose bin holds the pip ROCm SDK's hipcc, plus a
    system hipcc first on PATH and an expanded _rocm_sdk_devel package."""
    prefix = tmp_path / "venv"
    python = _executable(prefix / "bin" / "python")
    _executable(prefix / "bin" / "hipcc")
    system_bin = _executable(tmp_path / "opt" / "bin" / "hipcc").parent
    devel = tmp_path / "site" / "_rocm_sdk_devel"
    _executable(devel / "bin" / "hipconfig")

    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(system_bin))
    for var in ("AITER_JIT_DIR", "ROCM_HOME"):
        # setenv first so that teardown removes what the function sets.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    real_find_spec = importlib.util.find_spec
    devel_spec = importlib.machinery.ModuleSpec(
        "_rocm_sdk_devel", None, is_package=True
    )
    devel_spec.submodule_search_locations = [str(devel)]
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: (
            devel_spec if name == "_rocm_sdk_devel" else real_find_spec(name, *a)
        ),
    )
    return prefix, devel


def test_venv_sdk_compiles_and_the_jit_dir_is_per_venv(venv) -> None:
    prefix, devel = venv
    aiter_executor.prepare_aiter_environment()
    # Without these, AITER links against the hipDNN prefix, which has no
    # libamdhip64, or loads modules that another torch built.
    assert os.environ["ROCM_HOME"] == str(devel)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(prefix / "bin")
    assert os.environ["AITER_JIT_DIR"] == str(prefix / "aiter-jit")


def test_user_settings_win(venv, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path / "jit"))
    monkeypatch.setenv("ROCM_HOME", str(tmp_path / "rocm"))
    aiter_executor.prepare_aiter_environment()
    assert os.environ["AITER_JIT_DIR"] == str(tmp_path / "jit")
    assert os.environ["ROCM_HOME"] == str(tmp_path / "rocm")


def test_outside_a_venv_aiter_keeps_its_default_jit_dir(venv, monkeypatch) -> None:
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    aiter_executor.prepare_aiter_environment()
    assert "AITER_JIT_DIR" not in os.environ


def test_unexpanded_sdk_changes_no_toolchain(venv, monkeypatch) -> None:
    _, devel = venv
    (devel / "bin" / "hipconfig").unlink()
    path = os.environ["PATH"]
    aiter_executor.prepare_aiter_environment()
    assert "ROCM_HOME" not in os.environ
    assert os.environ["PATH"] == path


@pytest.fixture
def fake_aiter(monkeypatch):
    """``aiter.ops.mha`` and the Triton MHA module, recording each call."""
    calls = []

    def entry(name):
        return lambda *args, **kwargs: calls.append((name, args, kwargs))

    mha = types.ModuleType("aiter.ops.mha")
    for name in (
        "flash_attn_func",
        "mha_fwd",
        "fmha_v3_fwd",
        "fmha_fwd_bf16_opus_fwd",
        "mha_fwd_native_splitkv",
    ):
        setattr(mha, name, entry(name))
    triton_mha = types.ModuleType("aiter.ops.triton.attention.mha")
    triton_mha.flash_attn_func = entry("triton.flash_attn_func")
    ops = types.ModuleType("aiter.ops")
    ops.mha = mha
    for name, module in (
        ("aiter", types.ModuleType("aiter")),
        ("aiter.ops", ops),
        ("aiter.ops.mha", mha),
        ("aiter.ops.triton", types.ModuleType("aiter.ops.triton")),
        ("aiter.ops.triton.attention", types.ModuleType("aiter.ops.triton.attention")),
        ("aiter.ops.triton.attention.mha", triton_mha),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def _problem(causal, bottom_right):
    return DenseSdpa(
        data_type="bfloat16",
        batch=1,
        num_query_heads=8,
        num_kv_heads=8,
        seqlen_q=128,
        seqlen_kv=128,
        head_size_qk=128,
        head_size_v=128,
        causal=causal,
        bottom_right=bottom_right,
        scale=0.125,
        q_uid=1,
        k_uid=2,
        v_uid=3,
        o_uid=4,
    )


#: Each backend: the entry point it calls, the keyword that carries the mask,
#: and whether it writes the graph's output tensor.
_ENTRY_POINTS = {
    "auto": ("flash_attn_func", "causal", False),
    "ck": ("mha_fwd", "is_causal", True),
    "asm_v3": ("fmha_v3_fwd", "is_causal", True),
    "opus": ("fmha_fwd_bf16_opus_fwd", "causal", True),
    "native_splitkv": ("mha_fwd_native_splitkv", "causal", True),
    "triton": ("triton.flash_attn_func", "causal", False),
    "gluon": ("triton.flash_attn_func", "causal", False),
}


@pytest.mark.parametrize("backend", sorted(_ENTRY_POINTS))
@pytest.mark.parametrize(
    "causal, bottom_right, expected",
    [(False, False, False), (True, False, True), (False, True, True)],
    ids=["no-mask", "top-left", "bottom-right"],
)
def test_each_entry_point_gets_the_mask_scale_and_output(
    fake_aiter, backend, causal, bottom_right, expected
) -> None:
    q, k, v, o = object(), object(), object(), object()
    splits = 4 if backend == "native_splitkv" else 0
    launch = aiter_executor._launch_for(
        AiterBackend(backend, splits), _problem(causal, bottom_right), q, k, v, o
    )
    launch()

    entry, mask_key, writes_output = _ENTRY_POINTS[backend]
    [(name, args, kwargs)] = fake_aiter
    assert name == entry
    assert args == (q, k, v)
    assert kwargs[mask_key] is expected
    assert kwargs["softmax_scale"] == 0.125
    assert (kwargs.get("out") is o) == writes_output
    if backend == "native_splitkv":
        assert kwargs["num_splits"] == 4
    if backend in ("triton", "gluon"):
        assert kwargs["backend"] == backend
