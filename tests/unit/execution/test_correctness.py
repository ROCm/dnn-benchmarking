# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.correctness (per-engine output comparison)."""

import numpy as np
import pytest

from dnn_benchmarking.config.benchmark_config import SuiteConfig, ValidationConfig
from dnn_benchmarking.execution.correctness import check_correctness, tolerance_for
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.validation import ReferenceOutput


def _out(uid, data_type="float"):
    return TensorInfo(
        uid=uid,
        name=f"t{uid}",
        dims=[4],
        strides=[1],
        data_type=data_type,
        is_virtual=False,
        is_output=True,
    )


class _HostBM:
    """Buffer manager whose outputs live on the host only."""

    def __init__(self, outputs):
        self._outputs = outputs

    def get_output_tensor(self, uid):
        return None

    def get_output_data(self, uid):
        return self._outputs.get(uid)


def _config(rtol=None, atol=None):
    return SuiteConfig(
        validation=ValidationConfig(provider="pytorch", rtol=rtol, atol=atol)
    )


def _ref(data):
    return ReferenceOutput(data=np.asarray(data, dtype=np.float32), tensor_uid=0)


@pytest.mark.parametrize(
    "data_type, rtol, atol, expected",
    [
        ("bfloat16", None, None, (3e-2, 1e-3)),
        ("half", None, None, (1e-3, 1e-3)),
        ("float", None, None, (1e-5, 1e-6)),
        ("bfloat16", 0.5, None, (0.5, 0.5)),  # one explicit value sets both
    ],
)
def test_tolerance_by_dtype_and_override(data_type, rtol, atol, expected):
    assert tolerance_for(_config(rtol, atol), _out(1, data_type)) == expected


@pytest.mark.parametrize(
    "data_type, one_ulp_up",
    [
        ("fp8_e4m3", [1.125, 2.25, 0.5625, 4.5]),
        ("fp8_e4m3_fnuz", [1.125, 2.25, 0.5625, 4.5]),
        ("fp8_e5m2", [1.25, 2.5, 0.625, 5.0]),
        ("fp8_e5m2_fnuz", [1.25, 2.5, 0.625, 5.0]),
        ("fp8_e8m0", [2.0, 4.0, 1.0, 8.0]),
    ],
)
def test_fp8_output_one_ulp_apart_passes(data_type, one_ulp_up):
    """The two sides round different fp32 accumulators to fp8; one element
    landing on the neighbouring fp8 value is not a defect."""
    ref = {1: _ref([1.0, 2.0, 0.5, 4.0])}
    bm = _HostBM({1: np.array(one_ulp_up, np.float32)})

    c = check_correctness(bm, [_out(1, data_type)], ref, "pytorch", _config())

    assert c.tolerance_match is True


@pytest.mark.parametrize(
    "data_type, actual",
    [
        # Two ULP up from each reference element.
        ("fp8_e4m3", [1.25, 2.5, 0.625, 5.0]),
        ("fp8_e4m3_fnuz", [1.25, 2.5, 0.625, 5.0]),
        ("fp8_e5m2", [1.5, 3.0, 0.75, 6.0]),
        ("fp8_e5m2_fnuz", [1.5, 3.0, 0.75, 6.0]),
        # Two code steps down (4x smaller) and up (4x larger).
        ("fp8_e8m0", [0.25, 0.5, 0.125, 1.0]),
        ("fp8_e8m0", [4.0, 8.0, 2.0, 16.0]),
    ],
)
def test_fp8_output_two_ulp_apart_fails(data_type, actual):
    ref = {1: _ref([1.0, 2.0, 0.5, 4.0])}
    bm = _HostBM({1: np.array(actual, np.float32)})

    c = check_correctness(bm, [_out(1, data_type)], ref, "pytorch", _config())

    assert c.tolerance_match is False
    assert c.n_mismatch == 4


@pytest.mark.parametrize(
    "data_type, smallest_subnormal",
    [
        ("fp8_e4m3", 2**-9),
        ("fp8_e4m3_fnuz", 2**-10),
        ("fp8_e5m2", 2**-16),
        ("fp8_e5m2_fnuz", 2**-17),
    ],
)
def test_fp8_zero_reference_accepts_only_the_smallest_subnormal(
    data_type, smallest_subnormal
):
    ref = {1: _ref([0.0, 0.0, 0.0, 0.0])}
    actual = [smallest_subnormal, -smallest_subnormal, 2 * smallest_subnormal, 0.0]
    bm = _HostBM({1: np.array(actual, np.float32)})

    c = check_correctness(bm, [_out(1, data_type)], ref, "pytorch", _config())

    assert c.n_mismatch == 1


def test_aggregates_over_outputs_and_names_the_failing_output():
    ref = {1: _ref([1, 2, 3, 4]), 2: _ref([1, 1, 1, 1])}
    bm = _HostBM(
        {
            1: np.array([1, 2, 3, 5], np.float32),
            2: np.array([1, 1, 9, 5], np.float32),
        }
    )

    c = check_correctness(bm, [_out(1), _out(2)], ref, "pytorch", _config())

    assert c.tolerance_match is False
    assert (c.n_mismatch, c.n_total) == (3, 8)
    assert c.worst_output_uid == 2
    assert c.max_abs_diff == pytest.approx(8.0)
    assert "output 2" in c.error_message


def test_failing_output_outranks_a_passing_output_with_a_larger_diff():
    # SDPA-like mix: a bf16 O passes at a large diff, a float stat fails at a
    # small one. The verdict must name the failing output, not the largest diff.
    ref = {1: _ref([100, 100, 100, 100]), 2: _ref([1, 1, 1, 1])}
    bm = _HostBM(
        {
            1: np.array([100.5, 100, 100, 100], np.float32),
            2: np.array([1.01, 1, 1, 1], np.float32),
        }
    )

    c = check_correctness(bm, [_out(1, "bfloat16"), _out(2)], ref, "pytorch", _config())

    assert c.tolerance_match is False
    assert c.worst_output_uid == 2
    assert c.error_message.startswith("output 2:")
    assert c.max_abs_diff == pytest.approx(0.5)
    assert (c.rtol, c.atol) == (3e-2, 1e-3)


def test_passing_outputs_report_diffs_without_message():
    ref = {1: _ref([1, 2, 3, 4])}
    bm = _HostBM({1: np.array([1, 2, 3, 4.00001], np.float32)})

    c = check_correctness(bm, [_out(1)], ref, "pytorch", _config(rtol=1e-3))

    assert c.passed
    assert c.n_mismatch == 0 and c.n_total == 4
    assert c.error_message is None


@pytest.mark.parametrize(
    "refs, outputs, reason",
    [
        ({}, {1: np.zeros(4, np.float32)}, "did not produce output tensor UID 1"),
        ({1: _ref([0, 0, 0, 0])}, {}, "No output tensors to compare"),
    ],
)
def test_missing_side_is_a_failure_not_a_pass(refs, outputs, reason):
    config = _config(rtol=0.1, atol=0.2)
    c = check_correctness(_HostBM(outputs), [_out(1)], refs, "pytorch", config)

    assert c.explicitly_failed
    assert reason in c.error_message
    # The verdict reports the --rtol/--atol the run asked for.
    assert (c.rtol, c.atol) == (0.1, 0.2)


def test_read_failure_is_a_failure_not_a_crash():
    class FailingBM:
        def get_output_data(self, uid):
            raise RuntimeError("hipMemcpy failed")

    c = check_correctness(FailingBM(), [_out(1)], {1: _ref([0])}, "pytorch", _config())

    assert c.explicitly_failed
    assert c.error_message == "hipMemcpy failed"


def test_device_reference_is_compared_without_host_copy():
    torch = pytest.importorskip("torch")

    class DeviceBM:
        def get_output_tensor(self, uid):
            return torch.ones(4)

        def get_output_data(self, uid):
            raise AssertionError("host copy made despite a device reference")

    ref = {
        1: ReferenceOutput(
            data=np.ones(4, np.float32), tensor_uid=1, device_data=torch.ones(4)
        )
    }

    assert check_correctness(DeviceBM(), [_out(1)], ref, "pytorch", _config()).passed


def test_device_e8m0_compares_code_steps():
    torch = pytest.importorskip("torch")
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is None:
        pytest.skip("torch has no float8_e8m0fnu")
    expected = [1.0, 2.0, 0.5, 4.0]

    class DeviceBM:
        def get_output_tensor(self, uid):
            # One step down on two elements, two steps down on the others.
            return torch.tensor([0.5, 1.0, 0.125, 1.0]).to(e8m0)

    ref = {
        1: ReferenceOutput(
            data=np.asarray(expected, np.float32),
            tensor_uid=1,
            device_data=torch.tensor(expected).to(e8m0),
        )
    }

    c = check_correctness(DeviceBM(), [_out(1, "fp8_e8m0")], ref, "pytorch", _config())

    assert c.n_mismatch == 2


def test_device_compare_out_of_memory_falls_back_to_host(monkeypatch):
    """An OOM in the device compare is not a verdict; the host data decides."""
    torch = pytest.importorskip("torch")
    from dnn_benchmarking.execution import correctness

    host_compare = correctness.compare

    def compare(actual, expected, **kw):
        if isinstance(actual, torch.Tensor):
            raise torch.cuda.OutOfMemoryError("device compare")
        return host_compare(actual, expected, **kw)

    monkeypatch.setattr(correctness, "compare", compare)

    class DeviceBM:
        def get_output_tensor(self, uid):
            return torch.zeros(4)

        def get_output_data(self, uid):
            return np.ones(4, np.float32)

    ref = {
        1: ReferenceOutput(
            data=np.ones(4, np.float32), tensor_uid=1, device_data=torch.zeros(4)
        )
    }

    assert check_correctness(DeviceBM(), [_out(1)], ref, "pytorch", _config()).passed
