# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for PyTorchReferenceProvider.

These tests verify the PyTorch reference provider works correctly
without requiring GPU/hipDNN - they use stubbed input data and
verify the reference computations are correct.
"""

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import UnsupportedGraphError
from dnn_benchmarking.config.benchmark_config import ReferencePrecision
from dnn_benchmarking.validation import ReferenceProviderRegistry

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")


class TestPyTorchProviderAvailability:
    """Tests for PyTorchReferenceProvider availability."""

    def test_pytorch_provider_is_available(self) -> None:
        """Test that PyTorch provider is available when torch is installed."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        assert provider.is_available() is True

    def test_pytorch_provider_name(self) -> None:
        """Test PyTorch provider name."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        assert provider.name == "pytorch"


class TestPyTorchProviderConv:
    """Tests for convolution operations."""

    def test_conv_fwd_basic(self) -> None:
        """Test basic 2D convolution forward pass."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        # Create simple graph JSON for conv
        graph_json = {
            "tensors": [
                {"uid": 0, "name": "output", "dims": [1, 1, 3, 3]},
                {"uid": 1, "name": "input", "dims": [1, 1, 5, 5]},
                {"uid": 2, "name": "weight", "dims": [1, 1, 3, 3]},
            ],
            "nodes": [
                {
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"y_tensor_uid": 0},
                    "parameters": {
                        "pre_padding": [0, 0],
                        "stride": [1, 1],
                        "dilation": [1, 1],
                    },
                }
            ],
        }

        # Create input data
        input_x = np.ones((1, 1, 5, 5), dtype=np.float32)
        weight = np.ones((1, 1, 3, 3), dtype=np.float32)
        input_data = {1: input_x, 2: weight}

        # Compute reference
        outputs = provider.compute_reference(graph_json, input_data)

        # Check output
        assert 0 in outputs
        output = outputs[0].data

        # 3x3 conv with all-ones kernel on all-ones input gives 9.0 at each position
        assert output.shape == (1, 1, 3, 3)
        assert np.allclose(output, 9.0)

    def test_conv_fwd_with_padding(self) -> None:
        """Test 2D convolution with padding."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 0, "name": "output", "dims": [1, 1, 5, 5]},
                {"uid": 1, "name": "input", "dims": [1, 1, 5, 5]},
                {"uid": 2, "name": "weight", "dims": [1, 1, 3, 3]},
            ],
            "nodes": [
                {
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"y_tensor_uid": 0},
                    "parameters": {
                        "pre_padding": [1, 1],
                        "stride": [1, 1],
                        "dilation": [1, 1],
                    },
                }
            ],
        }

        input_x = np.ones((1, 1, 5, 5), dtype=np.float32)
        weight = np.ones((1, 1, 3, 3), dtype=np.float32)
        input_data = {1: input_x, 2: weight}

        outputs = provider.compute_reference(graph_json, input_data)

        # With same padding, output shape matches input shape
        assert outputs[0].data.shape == (1, 1, 5, 5)


class TestPyTorchProviderMatmul:
    """Tests for matrix multiplication operations."""

    def test_matmul_basic(self) -> None:
        """Test basic matrix multiplication."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "a", "dims": [2, 3]},
                {"uid": 2, "name": "b", "dims": [3, 4]},
                {"uid": 3, "name": "c", "dims": [2, 4]},
            ],
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                    "outputs": {"c_tensor_uid": 3},
                }
            ],
        }

        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32)
        input_data = {1: a, 2: b}

        outputs = provider.compute_reference(graph_json, input_data)

        assert 3 in outputs
        output = outputs[3].data

        # (2x3) @ (3x4) with all ones = 3.0 at each position
        assert output.shape == (2, 4)
        assert np.allclose(output, 3.0)

    def test_matmul_identity(self) -> None:
        """Test matmul with identity matrix."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "a", "dims": [3, 3]},
                {"uid": 2, "name": "b", "dims": [3, 3]},
                {"uid": 3, "name": "c", "dims": [3, 3]},
            ],
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                    "outputs": {"c_tensor_uid": 3},
                }
            ],
        }

        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        b = np.eye(3, dtype=np.float32)
        input_data = {1: a, 2: b}

        outputs = provider.compute_reference(graph_json, input_data)

        # A @ I = A
        assert np.allclose(outputs[3].data, a)

    @pytest.mark.parametrize(
        "tensor_dtype, io_data_type",
        [("bfloat16", None), ("unset", "bfloat16")],  # hipDNN fill_from_context
    )
    def test_matmul_bfloat16_uses_graph_dtype_and_returns_float32(
        self, tensor_dtype, io_data_type
    ) -> None:
        torch = pytest.importorskip("torch")
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph_json = {
            "io_data_type": io_data_type,
            "tensors": [
                {"uid": 1, "name": "a", "dims": [2, 3], "data_type": tensor_dtype},
                {"uid": 2, "name": "b", "dims": [3, 2], "data_type": tensor_dtype},
                {"uid": 3, "name": "c", "dims": [2, 2], "data_type": tensor_dtype},
            ],
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                    "outputs": {"c_tensor_uid": 3},
                }
            ],
        }
        a = np.array(
            [[1.00390625, -2.25, 0.33398438], [4.5, 1.0078125, -0.5]],
            dtype=np.float32,
        )
        b = np.array(
            [[0.25, -1.5], [2.0, 0.75], [-3.0, 1.25]],
            dtype=np.float32,
        )

        outputs = provider.compute_reference(graph_json, {1: a, 2: b})
        expected = (
            torch.matmul(
                torch.from_numpy(a).to(dtype=torch.bfloat16),
                torch.from_numpy(b).to(dtype=torch.bfloat16),
            )
            .float()
            .numpy()
        )

        assert outputs[3].data.dtype == np.float32
        np.testing.assert_array_equal(outputs[3].data, expected)


class TestPyTorchProviderPointwise:
    """Tests for pointwise operations."""

    def test_relu(self) -> None:
        """Test ReLU activation."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "input", "dims": [4]},
                {"uid": 2, "name": "output", "dims": [4]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "relu_fwd",
                        "in_0_tensor_uid": 1,
                        "relu_lower_clip": 0.0,
                        "relu_upper_clip": float("inf"),
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ],
        }

        input_x = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
        input_data = {1: input_x}

        outputs = provider.compute_reference(graph_json, input_data)

        expected = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        assert np.allclose(outputs[2].data, expected)

    def test_relu6(self) -> None:
        """Test ReLU6 (clipped ReLU)."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "input", "dims": [6]},
                {"uid": 2, "name": "output", "dims": [6]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "relu_fwd",
                        "in_0_tensor_uid": 1,
                        "relu_lower_clip": 0.0,
                        "relu_upper_clip": 6.0,
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ],
        }

        input_x = np.array([-1.0, 0.0, 3.0, 6.0, 7.0, 10.0], dtype=np.float32)
        input_data = {1: input_x}

        outputs = provider.compute_reference(graph_json, input_data)

        expected = np.array([0.0, 0.0, 3.0, 6.0, 6.0, 6.0], dtype=np.float32)
        assert np.allclose(outputs[2].data, expected)

    def test_add(self) -> None:
        """Test element-wise addition."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "a", "dims": [3]},
                {"uid": 2, "name": "b", "dims": [3]},
                {"uid": 3, "name": "c", "dims": [3]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "add",
                        "in_0_tensor_uid": 1,
                        "in_1_tensor_uid": 2,
                    },
                    "outputs": {"out_0_tensor_uid": 3},
                }
            ],
        }

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        input_data = {1: a, 2: b}

        outputs = provider.compute_reference(graph_json, input_data)

        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        assert np.allclose(outputs[3].data, expected)

    def test_mul(self) -> None:
        """Test element-wise multiplication."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "a", "dims": [3]},
                {"uid": 2, "name": "b", "dims": [3]},
                {"uid": 3, "name": "c", "dims": [3]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "mul",
                        "in_0_tensor_uid": 1,
                        "in_1_tensor_uid": 2,
                    },
                    "outputs": {"out_0_tensor_uid": 3},
                }
            ],
        }

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        input_data = {1: a, 2: b}

        outputs = provider.compute_reference(graph_json, input_data)

        expected = np.array([4.0, 10.0, 18.0], dtype=np.float32)
        assert np.allclose(outputs[3].data, expected)

    def test_tanh(self) -> None:
        """Test tanh activation."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "input", "dims": [3]},
                {"uid": 2, "name": "output", "dims": [3]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "tanh_fwd",
                        "in_0_tensor_uid": 1,
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ],
        }

        input_x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        input_data = {1: input_x}

        outputs = provider.compute_reference(graph_json, input_data)

        expected = np.tanh(input_x)
        assert np.allclose(outputs[2].data, expected)


class TestPyTorchProviderGraphSupport:
    """Tests for graph support checking."""

    def test_supports_graph_with_known_ops(self) -> None:
        """Test supports_graph returns True for supported operations."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "nodes": [
                {"type": "ConvolutionFwdAttributes"},
                {"type": "MatmulAttributes"},
                {"type": "PointwiseAttributes"},
            ]
        }

        assert provider.supports_graph(graph_json) is True

    def test_supports_graph_with_unknown_ops(self) -> None:
        """Test supports_graph returns False for unsupported operations."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "nodes": [
                {"type": "ConvolutionFwdAttributes"},
                {"type": "UnknownOperation"},
            ]
        }

        assert provider.supports_graph(graph_json) is False

    def test_get_unsupported_operations(self) -> None:
        """Test getting list of unsupported operations."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "nodes": [
                {"type": "ConvolutionFwdAttributes"},
                {"type": "UnknownOp1"},
                {"type": "UnknownOp2"},
            ]
        }

        unsupported = provider.get_unsupported_operations(graph_json)

        assert "UnknownOp1" in unsupported
        assert "UnknownOp2" in unsupported
        assert "ConvolutionFwdAttributes" not in unsupported

    def test_supported_operations(self) -> None:
        """Test getting set of supported operations."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        supported = provider.supported_operations()

        assert "ConvolutionFwdAttributes" in supported
        assert "MatmulAttributes" in supported
        assert "PointwiseAttributes" in supported
        assert "ConvolutionBwdAttributes" in supported
        assert "ConvolutionWrwAttributes" in supported
        assert "BatchnormAttributes" in supported
        assert "BatchnormInferenceAttributes" in supported
        assert "BatchnormInferenceAttributesVarianceExt" in supported
        assert "BatchnormBackwardAttributes" in supported
        assert "SdpaAttributes" in supported
        assert "SdpaBackwardAttributes" in supported
        assert "LayernormAttributes" in supported
        assert "RMSNormAttributes" in supported
        assert "RMSNormBackwardAttributes" in supported
        assert "ReductionAttributes" in supported
        assert "ResampleFwdAttributes" in supported


class TestPyTorchProviderErrors:
    """Tests for error handling."""

    def test_unsupported_operation_raises(self) -> None:
        """Test that unsupported operation raises ValueError."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {"nodes": [{"type": "UnsupportedOperation"}]}

        with pytest.raises(UnsupportedGraphError) as exc_info:
            provider.compute_reference(graph_json, {})

        assert "unsupported" in str(exc_info.value).lower()

    def test_unsupported_pointwise_operation_raises(self) -> None:
        """Test that unsupported pointwise operation raises ValueError."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "input", "dims": [3]},
                {"uid": 2, "name": "output", "dims": [3]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "unknown_pointwise_op",
                        "in_0_tensor_uid": 1,
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ],
        }

        input_data = {1: np.array([1.0, 2.0, 3.0])}

        with pytest.raises(UnsupportedGraphError) as exc_info:
            provider.compute_reference(graph_json, input_data)

        assert "unsupported" in str(exc_info.value).lower()


class TestPyTorchProviderNewOps:
    """Reference correctness for newly supported hipDNN operation types."""

    def test_matmul_batched_broadcast(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph_json = {
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                    "outputs": {"c_tensor_uid": 3},
                }
            ]
        }
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        b = np.arange(20, dtype=np.float32).reshape(1, 4, 5)
        outputs = provider.compute_reference(graph_json, {1: a, 2: b})

        np.testing.assert_allclose(
            outputs[3].data,
            torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy(),
        )

    def test_conv_dgrad_and_wgrad(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        common_params = {
            "conv_mode": "CROSS_CORRELATION",
            "pre_padding": [0, 0],
            "post_padding": [0, 0],
            "stride": [1, 1],
            "dilation": [1, 1],
        }
        dgrad_graph = {
            "tensors": [{"uid": 3, "dims": [1, 1, 4, 4]}],
            "nodes": [
                {
                    "type": "ConvolutionBwdAttributes",
                    "inputs": {"dy_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"dx_tensor_uid": 3},
                    "parameters": common_params,
                }
            ],
        }
        wrw_graph = {
            "tensors": [{"uid": 4, "dims": [1, 1, 2, 2]}],
            "nodes": [
                {
                    "type": "ConvolutionWrwAttributes",
                    "inputs": {"x_tensor_uid": 3, "dy_tensor_uid": 1},
                    "outputs": {"dw_tensor_uid": 4},
                    "parameters": common_params,
                }
            ],
        }
        dy = np.ones((1, 1, 3, 3), dtype=np.float32)
        w = np.ones((1, 1, 2, 2), dtype=np.float32)
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)

        dgrad = provider.compute_reference(dgrad_graph, {1: dy, 2: w})[3].data
        wrw = provider.compute_reference(wrw_graph, {1: dy, 3: x})[4].data

        np.testing.assert_allclose(
            dgrad,
            torch.nn.grad.conv2d_input(
                (1, 1, 4, 4), torch.from_numpy(w), torch.from_numpy(dy)
            ).numpy(),
        )
        np.testing.assert_allclose(
            wrw,
            torch.nn.grad.conv2d_weight(
                torch.from_numpy(x), (1, 1, 2, 2), torch.from_numpy(dy)
            ).numpy(),
        )

    def test_batchnorm_training_saved_and_running_stats(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph_json = {
            "nodes": [
                {
                    "type": "BatchnormAttributes",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "scale_tensor_uid": 2,
                        "bias_tensor_uid": 3,
                        "epsilon_tensor_uid": 4,
                        "prev_running_mean_tensor_uid": 5,
                        "prev_running_variance_tensor_uid": 6,
                        "momentum_tensor_uid": 7,
                    },
                    "outputs": {
                        "y_tensor_uid": 8,
                        "mean_tensor_uid": 9,
                        "inv_variance_tensor_uid": 10,
                        "next_running_mean_tensor_uid": 11,
                        "next_running_variance_tensor_uid": 12,
                    },
                }
            ]
        }
        x = np.array([[[[1.0, 3.0]]]], dtype=np.float32)
        input_data = {
            1: x,
            2: np.array([[[[1.0]]]], dtype=np.float32),
            3: np.array([[[[0.0]]]], dtype=np.float32),
            # Epsilon guards the division; zero is not a value the op ever
            # sees. A real 1e-5 moves y and inv_variance off their exact
            # values -- the three stats below do not depend on it.
            4: np.array([1e-5], dtype=np.float32),
            5: np.array([[[[10.0]]]], dtype=np.float32),
            6: np.array([[[[20.0]]]], dtype=np.float32),
            7: np.array([0.5], dtype=np.float32),
        }

        outputs = provider.compute_reference(graph_json, input_data)

        np.testing.assert_allclose(outputs[8].data, [[[[-1.0, 1.0]]]], rtol=1e-5)
        np.testing.assert_allclose(outputs[9].data, [[[[2.0]]]], rtol=1e-6)
        np.testing.assert_allclose(outputs[10].data, [[[[1.0]]]], rtol=1e-5)
        np.testing.assert_allclose(outputs[11].data, [[[[6.0]]]], rtol=1e-6)
        np.testing.assert_allclose(outputs[12].data, [[[[11.0]]]], rtol=1e-6)

    def test_batchnorm_inference_variance(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph_json = {
            "nodes": [
                {
                    "type": "BatchnormInferenceAttributesVarianceExt",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "mean_tensor_uid": 2,
                        "variance_tensor_uid": 3,
                        "scale_tensor_uid": 4,
                        "bias_tensor_uid": 5,
                        "epsilon_tensor_uid": 6,
                    },
                    "outputs": {"y_tensor_uid": 7},
                }
            ]
        }
        outputs = provider.compute_reference(
            graph_json,
            {
                1: np.array([[[[3.0]]]], dtype=np.float32),
                2: np.array([[[[1.0]]]], dtype=np.float32),
                3: np.array([[[[4.0]]]], dtype=np.float32),
                4: np.array([[[[2.0]]]], dtype=np.float32),
                5: np.array([[[[1.0]]]], dtype=np.float32),
                # Epsilon guards the division, so zero is not a configuration
                # the op ever sees; torch rejects it outright. A real 1e-5
                # moves y off an exact 3.0, hence the matching tolerance.
                6: np.array([1e-5], dtype=np.float32),
            },
        )

        np.testing.assert_allclose(outputs[7].data, [[[[3.0]]]], rtol=1e-5)

    def test_batchnorm_inference_mean_inv_variance(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph_json = {
            "nodes": [
                {
                    "type": "BatchnormInferenceAttributes",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "mean_tensor_uid": 2,
                        "inv_variance_tensor_uid": 3,
                        "scale_tensor_uid": 4,
                        "bias_tensor_uid": 5,
                    },
                    "outputs": {"y_tensor_uid": 6},
                }
            ]
        }
        outputs = provider.compute_reference(
            graph_json,
            {
                1: np.array([[[[3.0]]]], dtype=np.float32),
                2: np.array([[[[1.0]]]], dtype=np.float32),
                3: np.array([[[[0.5]]]], dtype=np.float32),
                4: np.array([[[[2.0]]]], dtype=np.float32),
                5: np.array([[[[1.0]]]], dtype=np.float32),
            },
        )

        np.testing.assert_allclose(outputs[6].data, [[[[3.0]]]], rtol=1e-6)

    def test_batchnorm_backward_outputs_expected_reductions(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph_json = {
            "nodes": [
                {
                    "type": "BatchnormBackwardAttributes",
                    "inputs": {
                        "dy_tensor_uid": 1,
                        "x_tensor_uid": 2,
                        "mean_tensor_uid": 3,
                        "inv_variance_tensor_uid": 4,
                        "scale_tensor_uid": 5,
                    },
                    "outputs": {
                        "dx_tensor_uid": 6,
                        "dscale_tensor_uid": 7,
                        "dbias_tensor_uid": 8,
                    },
                }
            ]
        }
        outputs = provider.compute_reference(
            graph_json,
            {
                1: np.array([[[[2.0, 4.0]]]], dtype=np.float32),
                2: np.array([[[[1.0, 3.0]]]], dtype=np.float32),
                3: np.array([[[[2.0]]]], dtype=np.float32),
                4: np.array([[[[1.0]]]], dtype=np.float32),
                5: np.array([[[[1.0]]]], dtype=np.float32),
            },
        )

        np.testing.assert_allclose(outputs[6].data, [[[[0.0, 0.0]]]], rtol=1e-6)
        np.testing.assert_allclose(outputs[7].data, [[[[2.0]]]], rtol=1e-6)
        np.testing.assert_allclose(outputs[8].data, [[[[6.0]]]], rtol=1e-6)

    def test_layernorm_reference_outputs_y_and_stats(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        scale = np.array([1.0, 0.5, 2.0, -1.0], dtype=np.float32)
        bias = np.array([0.0, 1.0, -0.5, 0.25], dtype=np.float32)
        graph_json = {
            "tensors": [
                {"uid": 5, "dims": [2, 3, 4], "data_type": "float"},
                {"uid": 6, "dims": [2, 3, 1], "data_type": "float"},
                {"uid": 7, "dims": [2, 3, 1], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "LayernormAttributes",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "scale_tensor_uid": 2,
                        "bias_tensor_uid": 3,
                        "epsilon_tensor_uid": 4,
                    },
                    "outputs": {
                        "y_tensor_uid": 5,
                        "mean_tensor_uid": 6,
                        "inv_variance_tensor_uid": 7,
                    },
                    "attributes": {"normalized_dim_count": 1},
                }
            ],
        }

        outputs = provider.compute_reference(
            graph_json,
            {1: x, 2: scale, 3: bias, 4: np.array([1e-5], dtype=np.float32)},
        )

        expected = torch.nn.functional.layer_norm(
            torch.from_numpy(x),
            (4,),
            torch.from_numpy(scale),
            torch.from_numpy(bias),
            1e-5,
        )
        np.testing.assert_allclose(outputs[5].data, expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(outputs[6].data, x.mean(axis=2, keepdims=True))
        np.testing.assert_allclose(
            outputs[7].data,
            1.0 / np.sqrt(x.var(axis=2, keepdims=True) + 1e-5),
            rtol=1e-6,
        )

    def test_rmsnorm_channel_reference_outputs_y_and_inv_rms(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        x = np.arange(1, 17, dtype=np.float32).reshape(1, 2, 2, 4)
        scale = np.array([[[[2.0]], [[0.5]]]], dtype=np.float32)
        bias = np.array([[[[0.25]], [[-1.0]]]], dtype=np.float32)
        graph_json = {
            "tensors": [
                {"uid": 5, "dims": [1, 2, 2, 4], "data_type": "float"},
                {"uid": 6, "dims": [1, 2, 1, 1], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "RMSNormAttributes",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "scale_tensor_uid": 2,
                        "epsilon_tensor_uid": 3,
                        "bias_tensor_uid": 4,
                    },
                    "outputs": {"y_tensor_uid": 5, "inv_rms_tensor_uid": 6},
                }
            ],
        }

        outputs = provider.compute_reference(
            graph_json,
            {1: x, 2: scale, 3: np.array([1e-5], dtype=np.float32), 4: bias},
        )

        inv = 1.0 / np.sqrt(np.mean(x * x, axis=(2, 3), keepdims=True) + 1e-5)
        np.testing.assert_allclose(outputs[6].data, inv, rtol=1e-6)
        np.testing.assert_allclose(outputs[5].data, x * inv * scale + bias, rtol=1e-6)

    def test_rmsnorm_backward_reference_matches_autograd(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        x_t = (
            torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
        ).requires_grad_()
        scale_t = torch.tensor([1.0, 0.5, 2.0, -1.0], requires_grad=True)
        dy_t = torch.linspace(-0.2, 0.3, steps=24).reshape(2, 3, 4)
        y_t = x_t * torch.rsqrt(x_t.square().mean(dim=2, keepdim=True) + 1e-5) * scale_t
        y_t.backward(dy_t)
        inv = torch.rsqrt(x_t.detach().square().mean(dim=2, keepdim=True) + 1e-5)

        graph_json = {
            "tensors": [
                {"uid": 5, "dims": [2, 3, 4], "data_type": "float"},
                {"uid": 6, "dims": [4], "data_type": "float"},
                {"uid": 7, "dims": [4], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "RMSNormBackwardAttributes",
                    "inputs": {
                        "dy_tensor_uid": 1,
                        "x_tensor_uid": 2,
                        "scale_tensor_uid": 3,
                        "inv_rms_tensor_uid": 4,
                    },
                    "outputs": {
                        "dx_tensor_uid": 5,
                        "dscale_tensor_uid": 6,
                        "dbias_tensor_uid": 7,
                    },
                }
            ],
        }

        outputs = provider.compute_reference(
            graph_json,
            {
                1: dy_t.numpy(),
                2: x_t.detach().numpy(),
                3: scale_t.detach().numpy(),
                4: inv.numpy(),
            },
        )

        np.testing.assert_allclose(outputs[5].data, x_t.grad.numpy(), rtol=1e-6)
        np.testing.assert_allclose(outputs[6].data, scale_t.grad.numpy(), rtol=1e-6)
        np.testing.assert_allclose(outputs[7].data, dy_t.sum(dim=(0, 1)).numpy())

    def test_reduction_and_resample_references(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        reduction_graph = {
            "tensors": [{"uid": 2, "dims": [2, 1], "data_type": "float"}],
            "nodes": [
                {
                    "type": "ReductionAttributes",
                    "inputs": {"in_tensor_uid": 1},
                    "outputs": {"out_tensor_uid": 2},
                    "attributes": {"mode": "ADD"},
                }
            ],
        }
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        reduction_outputs = provider.compute_reference(reduction_graph, {1: x})
        np.testing.assert_allclose(
            reduction_outputs[2].data, x.sum(axis=1, keepdims=True)
        )

        resample_graph = {
            "tensors": [{"uid": 2, "dims": [1, 1, 2], "data_type": "float"}],
            "nodes": [
                {
                    "type": "ResampleFwdAttributes",
                    "inputs": {"x_tensor_uid": 1},
                    "outputs": {"y_tensor_uid": 2},
                    "attributes": {
                        "pre_padding": [1],
                        "post_padding": [0],
                        "window": [3],
                        "stride": [2],
                        "resample_mode": "AVGPOOL_EXCLUDE_PADDING",
                        "padding_mode": "ZERO_PAD",
                    },
                }
            ],
        }
        pooled = provider.compute_reference(
            resample_graph,
            {1: np.arange(1, 6, dtype=np.float32).reshape(1, 1, 5)},
        )
        np.testing.assert_allclose(pooled[2].data, [[[1.5, 3.0]]])

    @pytest.mark.parametrize(
        "op_type,inputs,outputs",
        [
            (
                "BatchnormAttributes",
                {
                    "x_tensor_uid": 1,
                    "scale_tensor_uid": 2,
                    "bias_tensor_uid": 3,
                    "epsilon_tensor_uid": 4,
                    "peer_stats_tensor_uid": [99],
                },
                {"y_tensor_uid": 5},
            ),
            (
                "BatchnormBackwardAttributes",
                {
                    "dy_tensor_uid": 1,
                    "x_tensor_uid": 2,
                    "scale_tensor_uid": 3,
                    "peer_stats_tensor_uid": [99],
                },
                {
                    "dx_tensor_uid": 4,
                    "dscale_tensor_uid": 5,
                    "dbias_tensor_uid": 6,
                },
            ),
        ],
    )
    def test_batchnorm_peer_stats_rejected(
        self, op_type: str, inputs: dict, outputs: dict
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph_json = {
            "nodes": [
                {
                    "type": op_type,
                    "inputs": inputs,
                    "outputs": outputs,
                }
            ]
        }
        input_data = {
            1: np.array([[[[1.0, 3.0]]]], dtype=np.float32),
            2: np.array([[[[1.0]]]], dtype=np.float32),
            3: np.array([[[[0.0]]]], dtype=np.float32),
            4: np.array([1e-5], dtype=np.float32),
        }

        with pytest.raises(UnsupportedGraphError, match="peer statistics"):
            provider.compute_reference(graph_json, input_data)

    def test_sdpa_forward_matches_torch_and_returns_stats(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 2, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 3, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 4, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 6, "dims": [1, 1, 2, 1], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": 6},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})
        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k)
        v_t = torch.from_numpy(v)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, dropout_p=0.0, scale=1.0
        )
        expected_stats = torch.logsumexp(
            torch.matmul(q_t, k_t.transpose(-2, -1)),
            dim=-1,
            keepdim=True,
        )

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(outputs[6].data, expected_stats.numpy(), rtol=1e-6)
        assert outputs[6].data.shape == (1, 1, 2, 1)

    def test_sdpa_causal_forward_matches_torch_and_returns_masked_stats(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], dtype=np.float32)
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 1, 3, 2], "data_type": "float"},
                {"uid": 2, "dims": [1, 1, 3, 2], "data_type": "float"},
                {"uid": 3, "dims": [1, 1, 3, 2], "data_type": "float"},
                {"uid": 4, "dims": [1, 1, 3, 2], "data_type": "float"},
                {"uid": 6, "dims": [1, 1, 3, 1], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": 6},
                    "attributes": {"causal_mask": True, "dropout_probability": 0.0},
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})
        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k)
        v_t = torch.from_numpy(v)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, dropout_p=0.0, is_causal=True, scale=1.0
        )
        scores = torch.matmul(q_t, k_t.transpose(-2, -1))
        causal = torch.ones(3, 3, dtype=torch.bool).tril()
        expected_stats = torch.logsumexp(
            scores.masked_fill(~causal, float("-inf")), dim=-1, keepdim=True
        )

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(outputs[6].data, expected_stats.numpy(), rtol=1e-6)

    def test_sdpa_sliding_window_masks_output_and_stats(self) -> None:
        """A width-1 band (left/right bound 0) lets each query see only its own
        key: O == V and stats == q_i . k_i. Unmasked stats would differ."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)
        v = np.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], dtype=np.float32)
        dims = [1, 1, 3, 2]
        graph_json = {
            "tensors": [
                *(
                    {"uid": uid, "dims": dims, "data_type": "float"}
                    for uid in (1, 2, 3, 4)
                ),
                {"uid": 6, "dims": [1, 1, 3, 1], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": 6},
                    "attributes": {
                        "dropout_probability": 0.0,
                        "left_bound": 0,
                        "right_bound": 0,
                    },
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: q.copy(), 3: v})

        np.testing.assert_allclose(outputs[4].data, v, rtol=1e-6)
        np.testing.assert_allclose(outputs[6].data[..., 0], (q * q).sum(-1), rtol=1e-6)

    def test_sdpa_additive_attention_mask_matches_torch(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        mask = np.array([[[[0.0, -100.0], [0.0, 0.0]]]], dtype=np.float32)
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 2, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 3, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 4, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 5, "dims": [1, 1, 2, 2], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "attn_mask_tensor_uid": 5,
                    },
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v, 5: mask})
        expected = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(q),
            torch.from_numpy(k),
            torch.from_numpy(v),
            attn_mask=torch.from_numpy(mask),
            dropout_p=0.0,
            scale=1.0,
        )

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)

    def test_sdpa_gqa_matches_repeated_key_value_heads(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4) / 10.0
        k = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4) / 11.0
        v = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4) / 13.0
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 2, 2, 4], "data_type": "float"},
                {"uid": 2, "dims": [1, 1, 2, 4], "data_type": "float"},
                {"uid": 3, "dims": [1, 1, 2, 4], "data_type": "float"},
                {"uid": 4, "dims": [1, 2, 2, 4], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})
        expected = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(q),
            torch.from_numpy(k).repeat_interleave(2, dim=-3),
            torch.from_numpy(v).repeat_interleave(2, dim=-3),
            dropout_p=0.0,
            scale=1.0,
        )

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)

    def test_sdpa_backward_matches_autograd_with_consistent_stats(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        torch.manual_seed(0)
        scale = 1.0 / (8**0.5)
        q_t = torch.randn(2, 2, 4, 8, requires_grad=True)
        k_t = torch.randn(2, 2, 4, 8, requires_grad=True)
        v_t = torch.randn(2, 2, 4, 8, requires_grad=True)
        do_t = torch.randn(2, 2, 4, 8)
        scores = (q_t @ k_t.transpose(-2, -1)) * scale
        out = torch.softmax(scores, dim=-1) @ v_t
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        out.backward(do_t)

        graph_json = {
            "tensors": [
                {"uid": 10, "dims": [2, 2, 4, 8], "data_type": "float"},
                {"uid": 11, "dims": [2, 2, 4, 8], "data_type": "float"},
                {"uid": 12, "dims": [2, 2, 4, 8], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaBackwardAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "o_tensor_uid": 4,
                        "do_tensor_uid": 5,
                        "stats_tensor_uid": 6,
                    },
                    "outputs": {
                        "dq_tensor_uid": 10,
                        "dk_tensor_uid": 11,
                        "dv_tensor_uid": 12,
                    },
                    "attributes": {"attn_scale_value": scale},
                }
            ],
        }

        outputs = provider.compute_reference(
            graph_json,
            {
                1: q_t.detach().numpy(),
                2: k_t.detach().numpy(),
                3: v_t.detach().numpy(),
                4: out.detach().numpy(),
                5: do_t.numpy(),
                6: lse.detach().numpy(),
            },
        )

        np.testing.assert_allclose(
            outputs[10].data, q_t.grad.numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            outputs[11].data, k_t.grad.numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            outputs[12].data, v_t.grad.numpy(), rtol=1e-4, atol=1e-4
        )

    def test_sdpa_attention_scale_value_matches_torch(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {
                        "attn_scale_value": 0.25,
                        "dropout_probability": 0.0,
                    },
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})
        expected = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(q),
            torch.from_numpy(k),
            torch.from_numpy(v),
            dropout_p=0.0,
            scale=0.25,
        )

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)

    def test_sdpa_bottom_right_causal_aligns_to_sequence_end(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.zeros((1, 1, 2, 1), dtype=np.float32)
        k = np.zeros((1, 1, 4, 1), dtype=np.float32)
        v = np.array([[[[1.0], [2.0], [3.0], [4.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": 6},
                    "attributes": {
                        "left_bound": -1,
                        "right_bound": 0,
                        "diagonal_alignment": "BOTTOM_RIGHT",
                    },
                }
            ]
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})

        np.testing.assert_allclose(outputs[4].data, [[[[2.0], [2.5]]]])
        np.testing.assert_allclose(outputs[6].data, [[[[np.log(3.0)], [np.log(4.0)]]]])

    def test_sdpa_deprecated_causal_mask_overrides_bottom_right_alignment(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.zeros((1, 1, 2, 1), dtype=np.float32)
        k = np.zeros((1, 1, 4, 1), dtype=np.float32)
        v = np.array([[[[1.0], [2.0], [3.0], [4.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {
                        "causal_mask": True,
                        "diagonal_alignment": "BOTTOM_RIGHT",
                    },
                }
            ]
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})

        np.testing.assert_allclose(outputs[4].data, [[[[1.0], [1.5]]]])

    def test_sdpa_attention_sink_gqa_and_bottom_right_match_explicit_softmax(
        self,
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array(
            [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]]],
            dtype=np.float32,
        )
        k = np.array(
            [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]]],
            dtype=np.float32,
        )
        v = np.array(
            [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]],
            dtype=np.float32,
        )
        sink = np.log(np.array([2.0, 4.0], dtype=np.float32))
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "sink_token_tensor_uid": 5,
                    },
                    "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": 6},
                    "attributes": {
                        "attn_scale_value": 0.5,
                        "left_bound": -1,
                        "right_bound": 0,
                        "diagonal_alignment": "BOTTOM_RIGHT",
                    },
                }
            ]
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v, 5: sink})

        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k).repeat_interleave(2, dim=1)
        v_t = torch.from_numpy(v).repeat_interleave(2, dim=1)
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * 0.5
        bottom_right = torch.ones(2, 4, dtype=torch.bool).tril(diagonal=2)
        scores = scores.masked_fill(~bottom_right, float("-inf"))
        log_denom = torch.logaddexp(
            torch.logsumexp(scores, dim=-1, keepdim=True),
            torch.from_numpy(sink).reshape(1, 2, 1, 1),
        )
        expected = torch.exp(scores - log_denom) @ v_t

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(outputs[6].data, log_denom.numpy(), rtol=1e-6)

    def test_sdpa_bfloat16_uses_graph_dtype(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[0.10, 0.20], [0.30, 0.40]]]], dtype=np.float32)
        k = np.array([[[[0.50, 0.60], [0.70, 0.80]]]], dtype=np.float32)
        v = np.array([[[[0.90, 0.25], [0.125, 0.75]]]], dtype=np.float32)
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 1, 2, 2], "data_type": "bfloat16"},
                {"uid": 2, "dims": [1, 1, 2, 2], "data_type": "bfloat16"},
                {"uid": 3, "dims": [1, 1, 2, 2], "data_type": "bfloat16"},
                {"uid": 4, "dims": [1, 1, 2, 2], "data_type": "bfloat16"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})
        expected = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(q).to(torch.bfloat16),
            torch.from_numpy(k).to(torch.bfloat16),
            torch.from_numpy(v).to(torch.bfloat16),
            dropout_p=0.0,
            scale=1.0,
        ).to(torch.float32)

        assert outputs[4].data.dtype == np.float32
        np.testing.assert_array_equal(outputs[4].data, expected.numpy())

    @pytest.mark.parametrize(
        "optional_output",
        [
            "max_tensor_uid",
            "sum_exp_tensor_uid",
            "amax_s_tensor_uid",
            "amax_o_tensor_uid",
        ],
    )
    def test_sdpa_forward_rejects_unsupported_optional_outputs(
        self, optional_output: str
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4, optional_output: 5},
                    "attributes": {"dropout_probability": 0.0},
                }
            ]
        }

        with pytest.raises(UnsupportedGraphError, match=optional_output):
            provider.compute_reference(graph_json, {1: q, 2: k, 3: v})

    @pytest.mark.parametrize(
        "optional_input",
        [
            "block_mask_tensor_uid",
            "descale_q_tensor_uid",
            "descale_k_tensor_uid",
            "descale_v_tensor_uid",
            "descale_s_tensor_uid",
            "scale_s_tensor_uid",
            "scale_o_tensor_uid",
        ],
    )
    def test_sdpa_forward_rejects_unsupported_optional_inputs(
        self, optional_input: str
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        optional_input: 5,
                    },
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        with pytest.raises(UnsupportedGraphError, match=optional_input):
            provider.compute_reference(
                graph_json,
                {1: q, 2: q, 3: q, 5: np.array([1], dtype=np.int32)},
            )

    @pytest.mark.parametrize(
        "optional_input",
        ["page_table_k_tensor_uid", "page_table_v_tensor_uid"],
    )
    def test_sdpa_forward_paged_needs_both_tables_and_lengths(
        self, optional_input: str
    ) -> None:
        """Paged forward is served, but a half-specified paged graph is not: one
        page table without the other, or without seq_len_kv, cannot be gathered."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        optional_input: 5,
                    },
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        with pytest.raises(UnsupportedGraphError, match="page table|seq_len_kv"):
            provider.compute_reference(
                graph_json,
                {1: q, 2: q, 3: q, 5: np.array([1], dtype=np.int32)},
            )

    @pytest.mark.parametrize(
        "optional_input", ["seq_len_q_tensor_uid", "seq_len_kv_tensor_uid"]
    )
    def test_sdpa_forward_varlen_without_a_page_table_is_dense(
        self, optional_input: str
    ) -> None:
        """Sequence lengths alone do not make a graph paged. Without a page table
        there is nothing to gather, so the graph runs as the dense one it is."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        optional_input: 5,
                    },
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        result = provider.compute_reference(
            graph_json,
            {1: q, 2: q, 3: q, 5: np.array([2], dtype=np.int32)},
        )
        assert 4 in result

    @pytest.mark.parametrize(
        "attributes,match",
        [
            ({"alibi_mask": True}, "top-left causal"),
            (
                {"alibi_mask": True, "causal_mask_bottom_right": True},
                "bottom-right causal alignment",
            ),
            ({"padding_mask": True}, "padding masks"),
            ({"right_bound": 1}, "forward-looking band"),
            ({"left_bound": -5}, "neither unbounded nor a width"),
            (
                {"causal_mask": True, "causal_mask_bottom_right": True},
                "mutually exclusive",
            ),
            ({"diagonal_alignment": "CENTER"}, "diagonal_alignment"),
        ],
    )
    def test_sdpa_forward_rejects_unsupported_attributes(
        self, attributes: dict, match: str
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0, **attributes},
                }
            ],
        }

        with pytest.raises(UnsupportedGraphError, match=match):
            provider.compute_reference(graph_json, {1: q, 2: q, 3: q})

    @staticmethod
    def _sdpa_graph(q_dims, kv_dims, attributes, mask_dims=None, stats=False):
        def tensor(uid, dims):
            return {"uid": uid, "dims": dims, "data_type": "float"}

        inputs = {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3}
        tensors = [tensor(1, q_dims), tensor(2, kv_dims), tensor(3, kv_dims)]
        tensors.append(tensor(4, q_dims))
        outputs = {"o_tensor_uid": 4}
        if mask_dims is not None:
            inputs["attn_mask_tensor_uid"] = 5
            tensors.append(tensor(5, mask_dims))
        if stats:
            outputs["stats_tensor_uid"] = 6
            tensors.append(tensor(6, [*q_dims[:3], 1]))
        return {
            "tensors": tensors,
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": inputs,
                    "outputs": outputs,
                    "attributes": {
                        "dropout_probability": 0.0,
                        "attn_scale_value": 0.5,
                        **attributes,
                    },
                }
            ],
        }

    @staticmethod
    def _cudnn_alibi_slopes(num_heads: int) -> list:
        """Independent restatement of cuDNN's ``get_alibi_slope``."""
        base = 2 ** (num_heads.bit_length() - 1)
        slopes = [2.0 ** (-8.0 * (h + 1) / base) for h in range(base)]
        slopes += [2.0 ** (-4.0 * (2 * j + 1) / base) for j in range(num_heads - base)]
        return slopes

    @classmethod
    def _alibi_bias(cls, num_heads: int, q_len: int, kv_len: int):
        slopes = cls._cudnn_alibi_slopes(num_heads)
        bias = torch.zeros(1, num_heads, q_len, kv_len, dtype=torch.float32)
        for h in range(num_heads):
            for i in range(q_len):
                for j in range(kv_len):
                    bias[0, h, i, j] = (j - i) * slopes[h]
        return bias

    @staticmethod
    def _causal_additive(q_len: int, kv_len: int):
        mask = torch.zeros(q_len, kv_len, dtype=torch.float32)
        for i in range(q_len):
            for j in range(kv_len):
                if j > i:
                    mask[i, j] = float("-inf")
        return mask

    def test_sdpa_explicit_attn_mask_combined_with_causal_matches_band_plus_bias(
        self,
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        torch.manual_seed(3)
        q = torch.randn(1, 1, 3, 4)
        k = torch.randn(1, 1, 3, 4)
        v = torch.randn(1, 1, 3, 4)
        bias = torch.tensor(
            [[[[0.5, -1.0, 2.0], [1.5, 0.25, -0.75], [-0.5, 1.0, 0.125]]]]
        )
        graph_json = self._sdpa_graph(
            [1, 1, 3, 4], [1, 1, 3, 4], {"causal_mask": True}, mask_dims=[1, 1, 3, 3]
        )

        outputs = provider.compute_reference(
            graph_json, {1: q.numpy(), 2: k.numpy(), 3: v.numpy(), 5: bias.numpy()}
        )

        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias + self._causal_additive(3, 3), scale=0.5
        )
        # A bias-only result must differ, so a dropped causal band is visible.
        unmasked = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, scale=0.5
        )
        assert not np.allclose(expected.numpy(), unmasked.numpy(), rtol=1e-3)
        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-5)

    @pytest.mark.parametrize("heads", [8, 12])
    def test_alibi_slopes_match_hand_computed_values(self, heads: int) -> None:
        from dnn_benchmarking.execution.pytorch_ops.handlers.sdpa import _alibi_slopes

        if heads == 8:
            expected = [2.0**-e for e in range(1, 9)]
        else:
            expected = [2.0**-e for e in range(1, 9)] + [
                2.0**-0.5,
                2.0**-1.5,
                2.0**-2.5,
                2.0**-3.5,
            ]
        slopes = _alibi_slopes(heads, torch.device("cpu"), torch.float32)

        assert slopes.shape == (heads,)
        np.testing.assert_allclose(slopes.numpy(), expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "q_heads,kv_heads",
        [(3, 3), (5, 5), (4, 2)],
        ids=["h3", "h5", "gqa-4q-2kv"],
    )
    def test_sdpa_alibi_output_matches_explicit_attn_mask_with_same_bias(
        self, q_heads: int, kv_heads: int
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        torch.manual_seed(5)
        seq = 4
        q = torch.randn(1, q_heads, seq, 4)
        k = torch.randn(1, kv_heads, seq, 4)
        v = torch.randn(1, kv_heads, seq, 4)
        q_dims, kv_dims = [1, q_heads, seq, 4], [1, kv_heads, seq, 4]
        alibi_graph = self._sdpa_graph(
            q_dims, kv_dims, {"causal_mask": True, "alibi_mask": True}
        )
        bias_graph = self._sdpa_graph(
            q_dims,
            kv_dims,
            {"causal_mask": True},
            mask_dims=[1, q_heads, seq, seq],
        )
        bias = self._alibi_bias(q_heads, seq, seq)
        inputs = {1: q.numpy(), 2: k.numpy(), 3: v.numpy()}

        alibi_out = provider.compute_reference(alibi_graph, inputs)[4].data
        bias_out = provider.compute_reference(bias_graph, {**inputs, 5: bias.numpy()})[
            4
        ].data
        rep = q_heads // kv_heads
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k.repeat_interleave(rep, dim=1),
            v.repeat_interleave(rep, dim=1),
            attn_mask=bias + self._causal_additive(seq, seq),
            scale=0.5,
        )

        np.testing.assert_allclose(alibi_out, expected.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(bias_out, expected.numpy(), rtol=1e-5, atol=1e-6)
        # Without the bias the output differs, so ALiBi is actually applied.
        plain = torch.nn.functional.scaled_dot_product_attention(
            q,
            k.repeat_interleave(rep, dim=1),
            v.repeat_interleave(rep, dim=1),
            is_causal=True,
            scale=0.5,
        )
        assert not np.allclose(alibi_out, plain.numpy(), rtol=1e-3)

    def test_sdpa_alibi_with_top_left_sliding_window_matches_explicit_bias(
        self,
    ) -> None:
        """hipDNN's ALiBi support (isSupportedSdpaAlibi) allows any
        left_bound with right_bound == 0, wider than cuDNN's right_bound-only
        requirement; a top-left window is valid for both, and ALiBi-flagged
        and explicit-bias-plus-window graphs must agree, the same way the
        causal case above does."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        torch.manual_seed(17)
        heads, seq = 3, 6
        q = torch.randn(1, heads, seq, 4)
        k = torch.randn(1, heads, seq, 4)
        v = torch.randn(1, heads, seq, 4)
        q_dims = kv_dims = [1, heads, seq, 4]
        window_attrs = {"left_bound": 2, "right_bound": 0}
        alibi_graph = self._sdpa_graph(
            q_dims, kv_dims, {**window_attrs, "alibi_mask": True}
        )
        bias_graph = self._sdpa_graph(
            q_dims, kv_dims, window_attrs, mask_dims=[1, heads, seq, seq]
        )
        bias = self._alibi_bias(heads, seq, seq)
        inputs = {1: q.numpy(), 2: k.numpy(), 3: v.numpy()}

        alibi_out = provider.compute_reference(alibi_graph, inputs)[4].data
        bias_out = provider.compute_reference(bias_graph, {**inputs, 5: bias.numpy()})[
            4
        ].data

        np.testing.assert_allclose(alibi_out, bias_out, rtol=1e-5, atol=1e-6)
        # Without the window every key is visible, so the output must differ.
        unwindowed = provider.compute_reference(
            self._sdpa_graph(
                q_dims, kv_dims, {"causal_mask": True, "alibi_mask": True}
            ),
            inputs,
        )[4].data
        assert not np.allclose(alibi_out, unwindowed, rtol=1e-3)

    def test_sdpa_alibi_slope_follows_the_query_head_not_the_kv_head(self) -> None:
        """Two query heads share one KV head and read identical K/V and Q, so
        only the per-query-head slope can make their outputs differ."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        torch.manual_seed(7)
        q_row = torch.randn(1, 1, 4, 4)
        q = q_row.repeat(1, 4, 1, 1)
        k = torch.randn(1, 2, 4, 4)
        v = torch.randn(1, 2, 4, 4)
        graph_json = self._sdpa_graph(
            [1, 4, 4, 4], [1, 2, 4, 4], {"causal_mask": True, "alibi_mask": True}
        )

        out = provider.compute_reference(
            graph_json, {1: q.numpy(), 2: k.numpy(), 3: v.numpy()}
        )[4].data

        # Query heads 0 and 1 share KV head 0; so do 2 and 3 for KV head 1.
        assert not np.allclose(out[:, 0], out[:, 1], rtol=1e-3)
        assert not np.allclose(out[:, 2], out[:, 3], rtol=1e-3)

    def test_sdpa_alibi_non_square_uses_cudnn_form_not_flashattention_form(
        self,
    ) -> None:
        """With Sq != Skv under top-left causal, the FlashAttention form
        ``-slope * |i + Skv - Sq - j|`` shifts each row by a constant, so O is
        the same and only the log-sum-exp separates the two forms."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        torch.manual_seed(11)
        heads, sq, skv = 2, 2, 4
        q = torch.randn(1, heads, sq, 4)
        k = torch.randn(1, heads, skv, 4)
        v = torch.randn(1, heads, skv, 4)
        graph_json = self._sdpa_graph(
            [1, heads, sq, 4],
            [1, heads, skv, 4],
            {"causal_mask": True, "alibi_mask": True},
            stats=True,
        )

        outputs = provider.compute_reference(
            graph_json, {1: q.numpy(), 2: k.numpy(), 3: v.numpy()}
        )

        slopes = self._cudnn_alibi_slopes(heads)
        cudnn = self._alibi_bias(heads, sq, skv)
        flash = torch.zeros(1, heads, sq, skv)
        for h in range(heads):
            for i in range(sq):
                for j in range(skv):
                    flash[0, h, i, j] = -slopes[h] * abs(i + skv - sq - j)
        band = self._causal_additive(sq, skv)
        scores = torch.matmul(q, k.transpose(-2, -1)) * 0.5
        cudnn_lse = torch.logsumexp(scores + cudnn + band, dim=-1, keepdim=True)
        flash_lse = torch.logsumexp(scores + flash + band, dim=-1, keepdim=True)
        assert not np.allclose(cudnn_lse.numpy(), flash_lse.numpy(), rtol=1e-3)

        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=cudnn + band, scale=0.5
        )
        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-5)
        np.testing.assert_allclose(
            outputs[6].data, cudnn_lse.numpy(), rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize(
        "q_heads,kv_heads",
        [(3, 3), (4, 2)],
        ids=["h3", "gqa-4q-2kv"],
    )
    def test_sdpa_alibi_backward_matches_autograd(
        self, q_heads: int, kv_heads: int
    ) -> None:
        """GQA (q_heads != kv_heads) exercises that the ALiBi slope in the
        backward recomputation is indexed by the query head, not the KV head,
        the same way the forward path's gqa-4q-2kv case does."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        torch.manual_seed(13)
        seq, dim = 4, 4
        rep = q_heads // kv_heads
        q = torch.randn(1, q_heads, seq, dim, requires_grad=True)
        k = torch.randn(1, kv_heads, seq, dim, requires_grad=True)
        v = torch.randn(1, kv_heads, seq, dim, requires_grad=True)
        do = torch.randn(1, q_heads, seq, dim)
        bias = self._alibi_bias(q_heads, seq, seq) + self._causal_additive(seq, seq)
        scale = 0.5
        k_full = k.repeat_interleave(rep, dim=1)
        v_full = v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale + bias
        stats = torch.logsumexp(scores, dim=-1, keepdim=True)
        o = torch.matmul(torch.softmax(scores, dim=-1), v_full)
        dq, dk, dv = torch.autograd.grad(o, (q, k, v), do)

        q_dims = [1, q_heads, seq, dim]
        kv_dims = [1, kv_heads, seq, dim]
        tensor_list = [
            {"uid": uid, "dims": q_dims, "data_type": "float"} for uid in (1, 4, 5, 7)
        ]
        tensor_list += [
            {"uid": uid, "dims": kv_dims, "data_type": "float"} for uid in (2, 3, 8, 9)
        ]
        tensor_list.append(
            {"uid": 6, "dims": [1, q_heads, seq, 1], "data_type": "float"}
        )
        graph_json = {
            "tensors": tensor_list,
            "nodes": [
                {
                    "type": "SdpaBackwardAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "o_tensor_uid": 4,
                        "do_tensor_uid": 5,
                        "stats_tensor_uid": 6,
                    },
                    "outputs": {
                        "dq_tensor_uid": 7,
                        "dk_tensor_uid": 8,
                        "dv_tensor_uid": 9,
                    },
                    "attributes": {
                        "causal_mask": True,
                        "alibi_mask": True,
                        "attn_scale_value": scale,
                    },
                }
            ],
        }

        outputs = provider.compute_reference(
            graph_json,
            {
                1: q.detach().numpy(),
                2: k.detach().numpy(),
                3: v.detach().numpy(),
                4: o.detach().numpy(),
                5: do.numpy(),
                6: stats.detach().numpy(),
            },
        )

        for uid, grad in ((7, dq), (8, dk), (9, dv)):
            np.testing.assert_allclose(
                outputs[uid].data, grad.numpy(), rtol=1e-4, atol=1e-5
            )

    def test_sdpa_alibi_backward_correct_across_block_boundaries(
        self, monkeypatch
    ) -> None:
        """``_block_scores`` now builds the ALiBi bias from each block's own
        start offset instead of slicing a precomputed full-size tensor; a
        wrong offset would only show once a graph needs more than one
        block."""
        import dnn_benchmarking.execution.pytorch_ops.handlers.sdpa as sdpa_mod

        monkeypatch.setattr(sdpa_mod, "_SCORE_BLOCK_BYTES", 256)
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        torch.manual_seed(29)
        heads, seq, dim = 2, 9, 4
        q = torch.randn(1, heads, seq, dim, requires_grad=True)
        k = torch.randn(1, heads, seq, dim, requires_grad=True)
        v = torch.randn(1, heads, seq, dim, requires_grad=True)
        do = torch.randn(1, heads, seq, dim)
        bias = self._alibi_bias(heads, seq, seq) + self._causal_additive(seq, seq)
        scale = 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + bias
        stats = torch.logsumexp(scores, dim=-1, keepdim=True)
        o = torch.matmul(torch.softmax(scores, dim=-1), v)
        dq, dk, dv = torch.autograd.grad(o, (q, k, v), do)

        dims = [1, heads, seq, dim]
        tensor_list = [
            {"uid": uid, "dims": dims, "data_type": "float"}
            for uid in (1, 2, 3, 4, 5, 7, 8, 9)
        ]
        tensor_list.append({"uid": 6, "dims": [1, heads, seq, 1], "data_type": "float"})
        graph_json = {
            "tensors": tensor_list,
            "nodes": [
                {
                    "type": "SdpaBackwardAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "o_tensor_uid": 4,
                        "do_tensor_uid": 5,
                        "stats_tensor_uid": 6,
                    },
                    "outputs": {
                        "dq_tensor_uid": 7,
                        "dk_tensor_uid": 8,
                        "dv_tensor_uid": 9,
                    },
                    "attributes": {
                        "causal_mask": True,
                        "alibi_mask": True,
                        "attn_scale_value": scale,
                    },
                }
            ],
        }

        outputs = provider.compute_reference(
            graph_json,
            {
                1: q.detach().numpy(),
                2: k.detach().numpy(),
                3: v.detach().numpy(),
                4: o.detach().numpy(),
                5: do.numpy(),
                6: stats.detach().numpy(),
            },
        )

        for uid, grad in ((7, dq), (8, dk), (9, dv)):
            np.testing.assert_allclose(
                outputs[uid].data, grad.numpy(), rtol=1e-4, atol=1e-5
            )

    # Tensor UIDs of the SDPA backward dropout graphs below, after the
    # forward ones (``_Q`` to ``_DUMP``).
    _DO, _SCALE_INV, _DQ, _DK, _DV = 13, 14, 15, 16, 17

    @classmethod
    def _backward_dropout_graph(
        cls,
        *,
        mask=(2, 4, 5, 7),
        scale=(1,),
        scale_inv=(1,),
        probability=None,
        seed=False,
        offset=False,
        attn_mask=None,
        attributes=None,
        types=None,
    ):
        """An SDPA backward graph (B 2, Hq 4, Hkv 2, Sq 5, Skv 7, D 8) with an
        explicit dropout mask, its scale and ``dropout_scale_inv`` by default.

        ``mask``, ``scale``, ``scale_inv`` and ``attn_mask`` are dims (None
        leaves the tensor out). The probability is null by default.
        """
        q_dims, kv_dims = [2, 4, 5, 8], [2, 2, 7, 8]
        tensors = {
            _Q: q_dims,
            _K: kv_dims,
            _V: kv_dims,
            _O: q_dims,
            cls._DO: q_dims,
            _STATS: [2, 4, 5, 1],
            cls._DQ: q_dims,
            cls._DK: kv_dims,
            cls._DV: kv_dims,
        }
        inputs = {
            "q_tensor_uid": _Q,
            "k_tensor_uid": _K,
            "v_tensor_uid": _V,
            "o_tensor_uid": _O,
            "do_tensor_uid": cls._DO,
            "stats_tensor_uid": _STATS,
        }
        for key, uid, dims in (
            ("dropout_mask_tensor_uid", _DROP_MASK, mask),
            ("dropout_scale_tensor_uid", _DROP_SCALE, scale),
            ("dropout_scale_inv_tensor_uid", cls._SCALE_INV, scale_inv),
            ("attn_mask_tensor_uid", _ATTN_MASK, attn_mask),
            ("seed_tensor_uid", _SEED, (1,) if seed else None),
            ("offset_tensor_uid", _OFFSET, (1,) if offset else None),
        ):
            if dims is not None:
                inputs[key] = uid
                tensors[uid] = list(dims)
        types = {_SEED: "int64", _OFFSET: "int64", **(types or {})}
        return {
            "tensors": [
                {"uid": uid, "dims": dims, "data_type": types.get(uid, "float")}
                for uid, dims in tensors.items()
            ],
            "nodes": [
                {
                    "name": "sdpa_backward",
                    "type": "SdpaBackwardAttributes",
                    "inputs": inputs,
                    "outputs": {
                        "dq_tensor_uid": cls._DQ,
                        "dk_tensor_uid": cls._DK,
                        "dv_tensor_uid": cls._DV,
                    },
                    "attributes": {
                        "dropout_probability": probability,
                        "attn_scale_value": _ATTN_SCALE,
                        **(attributes or {}),
                    },
                }
            ],
        }

    @classmethod
    def _backward_dropout_inputs(cls, graph_json, seed):
        """``_dropout_inputs`` with ``dropout_scale_inv`` = 1/s."""
        inputs = _dropout_inputs(graph_json, seed)
        if cls._SCALE_INV in inputs:
            inputs[cls._SCALE_INV] = np.full(
                inputs[cls._SCALE_INV].shape, 1 / _KEEP_SCALE, dtype=np.float32
            )
        return inputs

    @staticmethod
    def _dropout_backward_reference(
        q, k, v, do, mask, s, inv, kind="none", attn_mask=None
    ):
        """float64 ``(O, stats, dQ, dK, dV)`` of hipDNN's dropout backward.

        ``O = (P * M * s) V`` with ``P`` the softmax before dropout, then
        ``D = rowsum(dO * O) * inv``, ``dP = (dO V^T) * M``,
        ``dS = s * P * (dP - D)`` and ``dV = (P * M * s)^T dO``; dQ and dK
        take the attention scale from ``dS``. The additive mask of ``kind``
        (none, causal, alibi) is built here from its definition.
        """
        q, k, v, do = (np.asarray(t, dtype=np.float64) for t in (q, k, v, do))
        mask = np.asarray(mask, dtype=np.float64)
        batch, hq, sq = q.shape[:3]
        hkv, skv = k.shape[1], k.shape[2]
        rep = hq // hkv
        k_full, v_full = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        i = np.arange(sq)[:, None]
        j = np.arange(skv)[None, :]
        causal = kind in ("causal", "alibi")
        additive = np.where(causal & (j > i), -np.inf, 0.0)
        if kind == "alibi":
            slopes = 2.0 ** (-8.0 * np.arange(1, hq + 1) / hq)  # power-of-two heads
            additive = additive + (j - i) * slopes[:, None, None]
        if attn_mask is not None:
            additive = additive + np.asarray(attn_mask, dtype=np.float64)
        scores = q @ k_full.swapaxes(-1, -2) * _ATTN_SCALE + additive
        lse = np.log(np.exp(scores).sum(axis=-1, keepdims=True))
        probs = np.exp(scores - lse)
        dropped = probs * mask * float(s)
        o = dropped @ v_full
        d = (do * o).sum(axis=-1, keepdims=True) * float(inv)
        dp = (do @ v_full.swapaxes(-1, -2)) * mask
        ds = float(s) * probs * (dp - d) * _ATTN_SCALE
        dk = ds.swapaxes(-1, -2) @ q
        dv = dropped.swapaxes(-1, -2) @ do
        return (
            o,
            lse,
            ds @ k_full,
            dk.reshape(batch, hkv, rep, skv, -1).sum(axis=2),
            dv.reshape(batch, hkv, rep, skv, -1).sum(axis=2),
        )

    def _assert_backward_closed_form(self, graph_json, seed, inv, kind="none"):
        """Run ``graph_json`` on the closed form's forward ``O`` and stats and
        compare dQ, dK and dV with ``_dropout_backward_reference``."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        inputs = self._backward_dropout_inputs(graph_json, seed)
        dropout = _DROP_MASK in inputs
        o, lse, *grads = self._dropout_backward_reference(
            inputs[_Q],
            inputs[_K],
            inputs[_V],
            inputs[self._DO],
            inputs[_DROP_MASK] if dropout else 1.0,
            _KEEP_SCALE if dropout else 1.0,
            inv,
            kind,
            attn_mask=inputs.get(_ATTN_MASK),
        )
        inputs[_O] = o.astype(np.float32)
        inputs[_STATS] = lse.astype(np.float32)

        out = provider.compute_reference(graph_json, inputs)

        for uid, grad in zip((self._DQ, self._DK, self._DV), grads):
            np.testing.assert_allclose(out[uid].data, grad, rtol=1e-4, atol=1e-4)

    def test_sdpa_backward_explicit_mask_matches_autograd(self) -> None:
        """With ``dropout_scale_inv`` = 1/s, hipDNN's dropout backward is the
        gradient of ``O = (P * M * s) V``. inv enters only ``D``, so dQ and dK
        catch a dropped inv. GQA (Hq 4, Hkv 2): the mask is per query head."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = self._backward_dropout_graph()
        inputs = self._backward_dropout_inputs(graph, seed=21)
        q, k, v = (
            torch.from_numpy(inputs[uid]).double().requires_grad_()
            for uid in (_Q, _K, _V)
        )
        mask = torch.from_numpy(inputs[_DROP_MASK]).double()
        scores = q @ k.repeat_interleave(2, dim=1).transpose(-2, -1) * _ATTN_SCALE
        dropped = torch.softmax(scores, dim=-1) * mask * float(_KEEP_SCALE)
        o = dropped @ v.repeat_interleave(2, dim=1)
        grads = torch.autograd.grad(
            o, (q, k, v), torch.from_numpy(inputs[self._DO]).double()
        )
        inputs[_O] = o.detach().float().numpy()
        inputs[_STATS] = (
            torch.logsumexp(scores, dim=-1, keepdim=True).detach().float().numpy()
        )

        out = provider.compute_reference(graph, inputs)

        for uid, grad in zip((self._DQ, self._DK, self._DV), grads):
            np.testing.assert_allclose(
                out[uid].data, grad.numpy(), rtol=1e-4, atol=1e-4
            )

    def test_sdpa_backward_scale_inv_absent_uses_one(self) -> None:
        """Without ``dropout_scale_inv``, ``D = rowsum(dO * O)``: inv is 1,
        not 1/s."""
        graph = self._backward_dropout_graph(scale_inv=None)
        self._assert_backward_closed_form(graph, seed=22, inv=1.0)

    @pytest.mark.parametrize("kind", ["causal_top_left", "additive_bias", "alibi"])
    def test_sdpa_backward_explicit_mask_composes_with_each_mask_kind(
        self, kind
    ) -> None:
        """The mask kinds the backward reference supports; sliding windows and
        bottom-right alignment stay declined in the backward."""
        attributes, ref_kind, bias, _ = _MASK_KINDS[kind]
        graph = self._backward_dropout_graph(
            attributes=attributes, attn_mask=[1, 4, 5, 7] if bias else None
        )
        self._assert_backward_closed_form(
            graph, seed=23, inv=1 / _KEEP_SCALE, kind=ref_kind
        )

    def test_sdpa_backward_probability_zero_is_no_dropout(self) -> None:
        """A non-null probability of 0 with no seed, offset or mask is the
        plain backward (R13)."""
        graph = self._backward_dropout_graph(
            mask=None, scale=None, scale_inv=None, probability=0.0
        )
        self._assert_backward_closed_form(graph, seed=24, inv=1.0)

    @pytest.mark.parametrize(
        "graph_kwargs",
        [
            pytest.param({"mask": None, "scale": None}, id="scale_inv_without_mask"),
            pytest.param({"types": {_SCALE_INV: "half"}}, id="scale_inv_not_float"),
            pytest.param({"scale_inv": [2]}, id="scale_inv_not_one_element"),
            pytest.param(
                {"mask": None, "scale": None, "scale_inv": None, "probability": 0.1},
                id="probability_nonzero",
            ),
            pytest.param(
                {
                    "mask": None,
                    "scale": None,
                    "scale_inv": None,
                    "probability": 0.0,
                    "seed": True,
                },
                id="probability_zero_with_seed",
            ),
            pytest.param(
                {
                    "mask": None,
                    "scale": None,
                    "scale_inv": None,
                    "probability": 0.0,
                    "offset": True,
                },
                id="probability_zero_with_offset",
            ),
            pytest.param({"probability": 0.0}, id="probability_zero_with_mask"),
        ],
    )
    def test_sdpa_backward_dropout_declines(self, graph_kwargs) -> None:
        """One case per backward R7 rule that hipDNN declines (rows 10-11).
        Each graph differs from a computable one in that rule only."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = self._backward_dropout_graph(**graph_kwargs)

        with pytest.raises(UnsupportedGraphError):
            provider.compute_reference(
                graph, self._backward_dropout_inputs(graph, seed=25)
            )

    @pytest.mark.parametrize("dtype", ["float", "bfloat16"])
    def test_sdpa_alibi_forward_correct_across_block_boundaries(
        self, dtype: str, monkeypatch
    ) -> None:
        """The forward builds its ALiBi mask (float32 or bf16 kernel input)
        and, below float32, the reference stats one query block at a time.
        A block's bias, ``attn_mask`` rows or window rows at a wrong offset
        show only once a graph needs more than one block."""
        import dnn_benchmarking.execution.pytorch_ops.handlers.sdpa as sdpa_mod

        monkeypatch.setattr(sdpa_mod, "_SCORE_BLOCK_BYTES", 512)
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        torch.manual_seed(31)
        heads, seq, dim, left_bound = 3, 9, 4, 3
        q = torch.randn(1, heads, seq, dim)
        k = torch.randn(1, heads, seq, dim)
        v = torch.randn(1, heads, seq, dim)
        extra = torch.randn(1, 1, seq, seq)
        graph_json = self._sdpa_graph(
            [1, heads, seq, dim],
            [1, heads, seq, dim],
            {"left_bound": left_bound, "right_bound": 0, "alibi_mask": True},
            mask_dims=[1, 1, seq, seq],
            stats=True,
        )
        for tensor in graph_json["tensors"]:
            if tensor["uid"] in (1, 2, 3, 4):
                tensor["data_type"] = dtype

        outputs = provider.compute_reference(
            graph_json, {1: q.numpy(), 2: k.numpy(), 3: v.numpy(), 5: extra.numpy()}
        )

        torch_dtype = torch.float32 if dtype == "float" else torch.bfloat16
        q, k, v = (t.to(torch_dtype).float() for t in (q, k, v))
        # Only the block offsets are under test, not the window width.
        width = sdpa_mod._sdpa_derive_mask(graph_json["nodes"][0])[1]
        band = sdpa_mod._diagonal_band_mask(
            seq, seq, width, bottom_right=False, device="cpu", dtype=torch.float32
        )
        mask = self._alibi_bias(heads, seq, seq) + extra + band
        scores = torch.matmul(q, k.transpose(-2, -1)) * 0.5 + mask
        expected_lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        expected_o = torch.matmul(torch.softmax(scores, dim=-1), v)
        o_tol = 1e-5 if dtype == "float" else 3e-2

        np.testing.assert_allclose(outputs[6].data, expected_lse.numpy(), atol=1e-4)
        np.testing.assert_allclose(
            outputs[4].data, expected_o.numpy(), rtol=o_tol, atol=o_tol
        )

    @pytest.mark.parametrize("fused", [False, True], ids=["cpu-kernel", "saved-lse"])
    def test_sdpa_alibi_bf16_stats_keep_fp32_precision_with_nonpow2_heads(
        self, fused: bool, monkeypatch
    ) -> None:
        """The SDPA kernel reads the ALiBi bias rounded to bf16, which loses
        log-sum-exp precision, worst with a non-power-of-two head count (H12),
        where the cuDNN slopes are not exact in bf16. The reference stats must
        come from the float32 bias, also when the kernel saves its own
        log-sum-exp (``fused``: a GPU Flash or efficient kernel, simulated
        with the log-sum-exp of the rounded bias)."""
        import dnn_benchmarking.execution.pytorch_ops.handlers.sdpa as sdpa_mod

        provider = ReferenceProviderRegistry.get_provider("pytorch")
        torch.manual_seed(23)
        heads, seq, dim = 12, 256, 8
        q = torch.randn(1, heads, seq, dim)
        k = torch.randn(1, heads, seq, dim)
        q_bf, k_bf = q.to(torch.bfloat16), k.to(torch.bfloat16)

        def tensor(uid, dtype="bfloat16"):
            return {"uid": uid, "dims": [1, heads, seq, dim], "data_type": dtype}

        graph_json = {
            "tensors": [
                tensor(1),
                tensor(2),
                tensor(3),
                tensor(4),
                {"uid": 6, "dims": [1, heads, seq, 1], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                    },
                    "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": 6},
                    "attributes": {
                        "dropout_probability": 0.0,
                        "attn_scale_value": 0.5,
                        "causal_mask": True,
                        "alibi_mask": True,
                    },
                }
            ],
        }

        bias = self._alibi_bias(heads, seq, seq)
        band = self._causal_additive(seq, seq)
        bf16_scores = torch.matmul(q_bf.float(), k_bf.float().transpose(-2, -1)) * 0.5
        exact_lse = torch.logsumexp(bf16_scores + bias + band, dim=-1, keepdim=True)
        rounded_bias = bias.to(torch.bfloat16).float()
        rounded_lse = torch.logsumexp(
            bf16_scores + rounded_bias + band, dim=-1, keepdim=True
        )
        if fused:
            monkeypatch.setattr(
                sdpa_mod, "fused_logsumexp", lambda o, q_len: rounded_lse
            )

        outputs = provider.compute_reference(
            graph_json,
            {
                1: q.numpy(),
                2: k.numpy(),
                3: torch.randn(1, heads, seq, dim).numpy(),
            },
        )

        # The pre-fix rounding error at H12 bf16 was about 3e-3; a correct
        # fp32 bias must land well under that bound, at the exact value.
        worst_if_rounded = (rounded_lse - exact_lse).abs().max().item()
        assert worst_if_rounded > 1e-3
        np.testing.assert_allclose(
            outputs[6].data, exact_lse.numpy(), atol=2e-4, rtol=0
        )

    def test_sdpa_alibi_is_declined_with_paged_attention(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "page_table_k_tensor_uid": 5,
                        "page_table_v_tensor_uid": 6,
                        "seq_len_kv_tensor_uid": 7,
                    },
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {
                        "dropout_probability": 0.0,
                        "causal_mask": True,
                        "alibi_mask": True,
                    },
                }
            ],
        }

        with pytest.raises(UnsupportedGraphError, match="paged"):
            provider.compute_reference(
                graph_json,
                {
                    1: q,
                    2: q,
                    3: q,
                    5: np.array([[0]], dtype=np.int32),
                    6: np.array([[0]], dtype=np.int32),
                    7: np.array([2], dtype=np.int32),
                },
            )


# UIDs of the SDPA dropout graphs below.
_Q, _K, _V, _O, _ATTN_MASK, _STATS = 1, 2, 3, 4, 5, 6
_DROP_MASK, _DROP_SCALE, _SINK, _SEED, _OFFSET, _DUMP = 7, 8, 9, 10, 11, 12
_ATTN_SCALE = 0.5
_KEEP_SCALE = np.float32(1 / 0.8)
_LEFT_BOUND = 2
_FULL = "full"
_INT_INPUTS = ("seed", "offset", "page_table", "seq_len")


def _dropout_graph(
    *,
    b=2,
    hq=4,
    hkv=2,
    sq=5,
    skv=7,
    d=8,
    mask=_FULL,
    scale=(1,),
    probability=None,
    seed=False,
    offset=False,
    dump=None,
    attn_mask=None,
    sink=False,
    attributes=None,
    types=None,
    extra_inputs=None,
):
    """An SDPA forward graph with an explicit dropout mask by default.

    ``mask``, ``scale``, ``dump`` and ``attn_mask`` are dims (None leaves the
    tensor out). The probability is null by default, the shipped form.
    """
    types = types or {}
    tensors = {
        _Q: [b, hq, sq, d],
        _K: [b, hkv, skv, d],
        _V: [b, hkv, skv, d],
        _O: [b, hq, sq, d],
        _STATS: [b, hq, sq, 1],
    }
    inputs = {"q_tensor_uid": _Q, "k_tensor_uid": _K, "v_tensor_uid": _V}
    outputs = {"o_tensor_uid": _O, "stats_tensor_uid": _STATS}
    if mask == _FULL:
        mask = [b, hq, sq, skv]
    for key, uid, dims in (
        ("dropout_mask_tensor_uid", _DROP_MASK, mask),
        ("dropout_scale_tensor_uid", _DROP_SCALE, scale),
        ("attn_mask_tensor_uid", _ATTN_MASK, attn_mask),
        ("sink_token_tensor_uid", _SINK, [hq] if sink else None),
        ("seed_tensor_uid", _SEED, [1] if seed else None),
        ("offset_tensor_uid", _OFFSET, [1] if offset else None),
    ):
        if dims is not None:
            inputs[key] = uid
            tensors[uid] = list(dims)
    if dump is not None:
        outputs["rng_dump_tensor_uid"] = _DUMP
        tensors[_DUMP] = list(dump)
    for key, (uid, dims) in (extra_inputs or {}).items():
        inputs[key] = uid
        tensors[uid] = list(dims)

    def data_type(uid, key):
        if uid in types:
            return types[uid]
        return "int64" if any(name in key for name in _INT_INPUTS) else "float"

    keys = {uid: key for key, uid in {**inputs, **outputs}.items()}
    return {
        "tensors": [
            {"uid": uid, "dims": dims, "data_type": data_type(uid, keys[uid])}
            for uid, dims in tensors.items()
        ],
        "nodes": [
            {
                "name": "sdpa",
                "type": "SdpaAttributes",
                "inputs": inputs,
                "outputs": outputs,
                "attributes": {
                    "dropout_probability": probability,
                    "attn_scale_value": _ATTN_SCALE,
                    **(attributes or {}),
                },
            }
        ],
    }


def _dropout_inputs(graph_json, seed):
    """Inputs of ``graph_json``: a 0/1 dropout mask at keep 0.8 with one zero
    column and one column of 1.5 (used as given), the scale 1/0.8, ints of 1
    for seed, offset and sequence tensors, normal floats for the rest."""
    rng = np.random.default_rng(seed)
    dims = {t["uid"]: t["dims"] for t in graph_json["tensors"]}
    data = {}
    for key, uid in graph_json["nodes"][0]["inputs"].items():
        shape = dims[uid]
        if uid == _DROP_MASK:
            mask = (rng.random(shape) < 0.8).astype(np.float32)
            if len(shape) == 4 and shape[-1] > 1:
                mask[..., 0] = 1.5
                mask[..., 1] = 0.0
            data[uid] = mask
        elif uid == _DROP_SCALE:
            data[uid] = np.full(shape, _KEEP_SCALE, dtype=np.float32)
        elif any(name in key for name in _INT_INPUTS):
            data[uid] = np.ones(shape, dtype=np.int64)
        else:
            data[uid] = rng.standard_normal(shape).astype(np.float32)
    return data


def dropout_reference(q, k, v, mask, s, kind="none", *, attn_mask=None, sink=None):
    """float64 ``O = (P * mask * s) V`` and the pre-dropout log-sum-exp.

    The additive mask of ``kind`` is built here from its definition. ``sink``
    adds one logit per query head to the softmax denominator.
    """
    q, k, v = (np.asarray(t, dtype=np.float64) for t in (q, k, v))
    hq, sq, skv = q.shape[1], q.shape[2], k.shape[2]
    k = np.repeat(k, hq // k.shape[1], axis=1)
    v = np.repeat(v, hq // v.shape[1], axis=1)
    i = np.arange(sq)[:, None]
    j = np.arange(skv)[None, :]
    visible = {
        "none": np.ones((sq, skv), dtype=bool),
        "causal": j <= i,
        "alibi": j <= i,
        "bottom_right": j <= i + (skv - sq),
        "window": (j <= i) & (j >= i - _LEFT_BOUND),
    }[kind]
    additive = np.where(visible, 0.0, -np.inf)
    if kind == "alibi":
        slopes = 2.0 ** (-8.0 * np.arange(1, hq + 1) / hq)  # power-of-two heads
        additive = additive + (j - i) * slopes[:, None, None]
    if attn_mask is not None:
        additive = additive + np.asarray(attn_mask, dtype=np.float64)
    scores = q @ k.swapaxes(-1, -2) * _ATTN_SCALE + additive
    lse = np.log(np.exp(scores).sum(axis=-1, keepdims=True))
    if sink is not None:
        lse = np.logaddexp(lse, np.asarray(sink, np.float64).reshape(1, -1, 1, 1))
    probs = np.exp(scores - lse)
    return (probs * np.asarray(mask, dtype=np.float64) * float(s)) @ v, lse


#: Mask kinds that compose with dropout (R14): graph attributes, the
#: ``dropout_reference`` kind, and whether the graph has a bias or sinks.
_MASK_KINDS = {
    "causal_top_left": ({"causal_mask": True}, "causal", False, False),
    "causal_bottom_right": ({"causal_mask_bottom_right": True}, "bottom_right", False, False),
    "sliding_window": ({"left_bound": _LEFT_BOUND, "right_bound": 0}, "window", False, False),
    "additive_bias": ({}, "none", True, False),
    "alibi": ({"causal_mask": True, "alibi_mask": True}, "alibi", False, False),
    "sinks": ({}, "none", False, True),
}  # fmt: skip


def _composed(kind, **graph_kwargs):
    """A dropout graph with the mask kind ``kind`` and its reference kwargs."""
    attributes, ref_kind, bias, sink = _MASK_KINDS[kind]
    graph = _dropout_graph(
        attributes=attributes,
        attn_mask=[1, 4, 5, 7] if bias else None,  # [1, Hq, Sq, Skv]
        sink=sink,
        **graph_kwargs,
    )
    return graph, ref_kind, bias, sink


class TestSdpaExplicitDropout:
    """Explicit-mask dropout in the SDPA forward reference (R6, R7, R14, R15)."""

    @pytest.mark.parametrize(
        "mask_dims",
        [[2, 4, 5, 7], [1, 4, 5, 7], [2, 1, 5, 7], [1, 1, 5, 7]],
        ids=["B,Hq", "1,Hq", "B,1", "1,1"],
    )
    def test_sdpa_explicit_mask_forward_matches_closed_form(self, mask_dims) -> None:
        """GQA (Hq 4, Hkv 2): the mask head index is the query head. The
        stats are the log-sum-exp before dropout."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _dropout_graph(mask=mask_dims)
        inputs = _dropout_inputs(graph, seed=11)

        out = provider.compute_reference(graph, inputs)

        o, lse = dropout_reference(
            inputs[_Q], inputs[_K], inputs[_V], inputs[_DROP_MASK], _KEEP_SCALE
        )
        np.testing.assert_allclose(out[_O].data, o, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out[_STATS].data, lse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("kind", list(_MASK_KINDS))
    def test_sdpa_explicit_mask_composes_with_each_mask_kind(self, kind) -> None:
        """With sinks, P is normalised with the sink before ``M * s``."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph, ref_kind, bias, sink = _composed(kind)
        inputs = _dropout_inputs(graph, seed=12)

        out = provider.compute_reference(graph, inputs)

        o, lse = dropout_reference(
            inputs[_Q],
            inputs[_K],
            inputs[_V],
            inputs[_DROP_MASK],
            _KEEP_SCALE,
            ref_kind,
            attn_mask=inputs[_ATTN_MASK] if bias else None,
            sink=inputs[_SINK] if sink else None,
        )
        np.testing.assert_allclose(out[_O].data, o, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out[_STATS].data, lse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "mask_dims", [[2, 4, 9, 11], [1, 1, 9, 11]], ids=["B,Hq", "1,1"]
    )
    def test_sdpa_explicit_mask_blocks_match_one_block(
        self, mask_dims, monkeypatch
    ) -> None:
        """Nine query rows in blocks of two: each block must take its own rows
        of the dropout mask and of the bias. Every query sees a key."""
        from dnn_benchmarking.execution.pytorch_ops.handlers import sdpa as sdpa_mod

        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _dropout_graph(
            sq=9,
            skv=11,
            mask=mask_dims,
            attn_mask=[1, 4, 9, 11],
            attributes={"causal_mask_bottom_right": True},
        )
        inputs = _dropout_inputs(graph, seed=13)
        one_block = provider.compute_reference(graph, inputs)

        # Scores of one query row: 2 batches x 4 heads x 11 keys x 4 bytes.
        monkeypatch.setattr(sdpa_mod, "_SCORE_BLOCK_BYTES", 2 * 2 * 4 * 11 * 4)
        assert len(sdpa_mod._query_blocks(torch.zeros(2, 4, 9, 8), 11)) == 5
        blocked = provider.compute_reference(graph, inputs)

        o, lse = dropout_reference(
            inputs[_Q],
            inputs[_K],
            inputs[_V],
            inputs[_DROP_MASK],
            _KEEP_SCALE,
            "bottom_right",
            attn_mask=inputs[_ATTN_MASK],
        )
        for uid, expected in ((_O, o), (_STATS, lse)):
            np.testing.assert_allclose(
                blocked[uid].data, one_block[uid].data, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                blocked[uid].data, expected, rtol=1e-5, atol=1e-5
            )

    @pytest.mark.parametrize(
        "extra_inputs,feature",
        [
            (
                {
                    "page_table_k_tensor_uid": (13, [2, 1]),
                    "page_table_v_tensor_uid": (14, [2, 1]),
                    "seq_len_kv_tensor_uid": (15, [2]),
                },
                "page_table",
            ),
            ({"seq_len_q_tensor_uid": (15, [2])}, "seq_len_q"),
            ({"seq_len_kv_tensor_uid": (15, [2])}, "seq_len_kv"),
        ],
        ids=["paged", "ragged_q", "ragged_kv"],
    )
    def test_sdpa_dropout_declined_for_paged_and_ragged_kv(
        self, extra_inputs, feature
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _dropout_graph(extra_inputs=extra_inputs)

        with pytest.raises(UnsupportedGraphError, match=feature):
            provider.compute_reference(graph, _dropout_inputs(graph, seed=14))

    @pytest.mark.parametrize(
        "graph_kwargs,match",
        [
            pytest.param(
                {"probability": 0.0}, None, id="mask_and_scale_with_probability"
            ),
            pytest.param({"seed": True}, None, id="mask_and_scale_with_seed"),
            pytest.param({"offset": True}, None, id="mask_and_scale_with_offset"),
            # The message names the given key and the missing one.
            pytest.param(
                {"scale": None},
                "dropout_mask_tensor_uid.*dropout_scale_tensor_uid",
                id="mask_without_scale",
            ),
            pytest.param(
                {"mask": None},
                "dropout_scale_tensor_uid.*dropout_mask_tensor_uid",
                id="scale_without_mask",
            ),
            pytest.param({"types": {_DROP_SCALE: "half"}}, None, id="scale_not_float"),
            pytest.param({"scale": [2]}, None, id="scale_not_one_element"),
            pytest.param({"types": {_DROP_MASK: "half"}}, None, id="mask_not_float"),
            pytest.param({"mask": [4, 5, 7]}, None, id="mask_not_rank_4"),
            # Rank 1 with a first dim that would pass the dim0 rule.
            pytest.param({"mask": [1]}, None, id="mask_rank_1"),
            pytest.param({"mask": [3, 4, 5, 7]}, None, id="mask_dim0_not_b_or_1"),
            # Hkv heads: the mask is indexed by the query head.
            pytest.param({"mask": [2, 2, 5, 7]}, None, id="mask_dim1_not_hq_or_1"),
            pytest.param({"mask": [2, 4, 1, 7]}, None, id="mask_dim2_not_sq"),
            pytest.param({"mask": [2, 4, 5, 1]}, None, id="mask_dim3_not_skv"),
        ],
    )
    def test_sdpa_dropout_declines(self, graph_kwargs, match) -> None:
        """One case per R7 rule that hipDNN declines (rows 1-4). Each graph
        differs from a computable one in that rule only."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _dropout_graph(**graph_kwargs)

        with pytest.raises(UnsupportedGraphError, match=match):
            provider.compute_reference(graph, _dropout_inputs(graph, seed=15))


#: f32(1) / (f32(1) - f32(p)), the Philox scale as hipDNN evaluates it.
_P = 0.3
_PHILOX_SCALE = np.float32(1) / (np.float32(1) - np.float32(_P))
_DUMP_DIMS = [2, 4, 5, 7]  # [B, Hq, Sq, Skv]


def _philox_graph(**graph_kwargs):
    """A Philox dropout forward with seed, offset and an rng_dump output."""
    graph_kwargs.setdefault("mask", None)
    graph_kwargs.setdefault("scale", None)
    graph_kwargs.setdefault("probability", _P)
    graph_kwargs.setdefault("seed", True)
    graph_kwargs.setdefault("offset", True)
    graph_kwargs.setdefault("dump", _DUMP_DIMS)
    return _dropout_graph(**graph_kwargs)


def _philox_dump(seed, dims=_DUMP_DIMS):
    """A 0/1 dump at keep 0.7 with both values in every row."""
    dump = (np.random.default_rng(seed).random(dims) < 0.7).astype(np.float32)
    if len(dims) == 4 and dims[-1] > 1:
        dump[..., 0] = 0.0
        dump[..., 1] = 1.0
    return dump


class TestSdpaPhiloxDropout:
    """Philox dropout through the engine's rng_dump (R0, R7, R8, R9, R14, R18,
    R23). A call passes the engine row's dump with ``philox_dumps``."""

    def test_philox_forward_applies_the_given_dump_with_one_over_one_minus_p(
        self,
    ) -> None:
        """p 0.3, not 0.5, so a 1/p scale differs from 1/(1-p). The stats are
        the log-sum-exp before dropout. The dump is not a reference output."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _philox_graph()
        inputs = _dropout_inputs(graph, seed=21)
        dump = _philox_dump(22)

        out = provider.compute_reference(graph, inputs, philox_dumps={_DUMP: dump})

        o, lse = dropout_reference(
            inputs[_Q], inputs[_K], inputs[_V], dump, _PHILOX_SCALE
        )
        np.testing.assert_allclose(out[_O].data, o, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out[_STATS].data, lse, rtol=1e-5, atol=1e-5)
        assert _DUMP not in out

    @pytest.mark.parametrize("kind", list(_MASK_KINDS))
    def test_philox_forward_composes_with_each_mask_kind(self, kind) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph, ref_kind, bias, sink = _composed(
            kind,
            mask=None,
            scale=None,
            probability=_P,
            seed=True,
            offset=True,
            dump=_DUMP_DIMS,
        )
        inputs = _dropout_inputs(graph, seed=23)
        dump = _philox_dump(24)

        out = provider.compute_reference(graph, inputs, philox_dumps={_DUMP: dump})

        o, lse = dropout_reference(
            inputs[_Q],
            inputs[_K],
            inputs[_V],
            dump,
            _PHILOX_SCALE,
            ref_kind,
            attn_mask=inputs[_ATTN_MASK] if bias else None,
            sink=inputs[_SINK] if sink else None,
        )
        np.testing.assert_allclose(out[_O].data, o, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out[_STATS].data, lse, rtol=1e-5, atol=1e-5)

    def test_philox_reference_uses_the_dump_it_is_given(self) -> None:
        """Two engine rows with different dumps: each output follows its own
        dump, so the reference keeps no mask from an earlier call."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _philox_graph()
        inputs = _dropout_inputs(graph, seed=25)
        dumps = [_philox_dump(26), _philox_dump(27)]

        outputs = [
            provider.compute_reference(graph, inputs, philox_dumps={_DUMP: dump})
            for dump in dumps
        ]

        for out, dump in zip(outputs, dumps):
            o, _ = dropout_reference(
                inputs[_Q], inputs[_K], inputs[_V], dump, _PHILOX_SCALE
            )
            np.testing.assert_allclose(out[_O].data, o, rtol=1e-5, atol=1e-5)
        assert not np.allclose(outputs[0][_O].data, outputs[1][_O].data, atol=1e-3)

    def test_philox_dump_in_the_input_tensors_is_ignored(self) -> None:
        """A buffer manager can hand the zeroed rng_dump buffer over with the
        inputs. Only ``philox_dumps`` is the dump."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _philox_graph()
        inputs = _dropout_inputs(graph, seed=28)
        inputs[_DUMP] = np.zeros(_DUMP_DIMS, dtype=np.float32)
        dump = _philox_dump(29)

        with pytest.raises(UnsupportedGraphError, match="rng_dump"):
            provider.compute_reference(graph, inputs)
        out = provider.compute_reference(graph, inputs, philox_dumps={_DUMP: dump})

        o, _ = dropout_reference(
            inputs[_Q], inputs[_K], inputs[_V], dump, _PHILOX_SCALE
        )
        np.testing.assert_allclose(out[_O].data, o, rtol=1e-5, atol=1e-5)

    def test_philox_without_scope_declines_naming_rng_dump(self) -> None:
        """PyTorch rows of Philox graphs stay declined (R18): they compile
        outside a ``philox_dumps`` call, before and after one."""
        from dnn_benchmarking.execution import pytorch_ops

        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _philox_graph()
        inputs = _dropout_inputs(graph, seed=30)

        with pytest.raises(UnsupportedGraphError, match="rng_dump"):
            pytorch_ops.compile_graph(graph)
        provider.compute_reference(
            graph, inputs, philox_dumps={_DUMP: _philox_dump(31)}
        )
        with pytest.raises(UnsupportedGraphError, match="rng_dump"):
            pytorch_ops.compile_graph(graph)

    @pytest.mark.parametrize("ones", [False, True], ids=["all_zero", "all_one"])
    def test_philox_p_zero_with_dump_uses_the_dump_as_is(self, ones) -> None:
        """At p 0 the scale is 1: an all-zero dump gives O = 0, an all-one dump
        gives the forward without dropout."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _philox_graph(probability=0.0)
        inputs = _dropout_inputs(graph, seed=32)
        dump = np.full(_DUMP_DIMS, 1.0 if ones else 0.0, dtype=np.float32)

        out = provider.compute_reference(graph, inputs, philox_dumps={_DUMP: dump})

        if ones:
            o, _ = dropout_reference(inputs[_Q], inputs[_K], inputs[_V], dump, 1.0)
            np.testing.assert_allclose(out[_O].data, o, rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_array_equal(out[_O].data, 0.0)

    @pytest.mark.parametrize(
        "graph_kwargs,dump_dims,match",
        [
            pytest.param(
                {"probability": None},
                _DUMP_DIMS,
                None,
                id="rng_dump_without_probability",
            ),
            pytest.param(
                {"probability": None, "dump": None, "offset": False},
                None,
                None,
                id="seed_without_probability",
            ),
            pytest.param(
                {"probability": None, "dump": None, "seed": False},
                None,
                None,
                id="offset_without_probability",
            ),
            pytest.param(
                {
                    "probability": None,
                    "seed": False,
                    "offset": False,
                    "mask": _FULL,
                    "scale": (1,),
                },
                _DUMP_DIMS,
                None,
                id="rng_dump_with_mask_and_scale",
            ),
            pytest.param({"seed": False}, _DUMP_DIMS, None, id="rng_dump_without_seed"),
            pytest.param(
                {"offset": False}, _DUMP_DIMS, None, id="rng_dump_without_offset"
            ),
            pytest.param({"probability": float("nan")}, _DUMP_DIMS, None, id="p_nan"),
            pytest.param({"probability": -0.1}, _DUMP_DIMS, None, id="p_below_0"),
            pytest.param({"probability": 1.0}, _DUMP_DIMS, None, id="p_1"),
            pytest.param({"probability": 1.5}, _DUMP_DIMS, None, id="p_above_1"),
            pytest.param(
                {"types": {_DUMP: "half"}}, _DUMP_DIMS, None, id="rng_dump_not_float"
            ),
            pytest.param(
                {"dump": [1, 4, 5, 7]}, [1, 4, 5, 7], None, id="rng_dump_dim0_not_b"
            ),
            # Hkv heads: the dump is indexed by the query head.
            pytest.param(
                {"dump": [2, 2, 5, 7]}, [2, 2, 5, 7], None, id="rng_dump_dim1_not_hq"
            ),
            pytest.param(
                {"dump": [2, 4, 1, 7]}, [2, 4, 1, 7], None, id="rng_dump_dim2_not_sq"
            ),
            pytest.param(
                {"dump": [2, 4, 5, 1]}, [2, 4, 5, 1], None, id="rng_dump_dim3_not_skv"
            ),
            pytest.param(
                {"dump": None}, None, "rng_dump", id="philox_without_rng_dump"
            ),
            pytest.param(
                {"extra_inputs": {"descale_q_tensor_uid": (16, [1])}},
                _DUMP_DIMS,
                None,
                id="fp8_scaling",
            ),
            pytest.param(
                {"extra_inputs": {"seq_len_kv_tensor_uid": (15, [2])}},
                _DUMP_DIMS,
                "seq_len_kv",
                id="ragged_kv",
            ),
        ],
    )
    def test_philox_declines(self, graph_kwargs, dump_dims, match) -> None:
        """One case per R7 rule for Philox (rows 5-9). The dump is in scope,
        so only the rule can decline the graph."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _philox_graph(**graph_kwargs)
        dumps = {} if dump_dims is None else {_DUMP: _philox_dump(33, dump_dims)}

        with pytest.raises(UnsupportedGraphError, match=match):
            provider.compute_reference(
                graph, _dropout_inputs(graph, seed=34), philox_dumps=dumps
            )

    def test_sdpa_probability_zero_without_dump_is_no_dropout(self) -> None:
        """p 0 (not null) with seed and offset and no rng_dump, mask or scale:
        no dropout, and seed and offset are not read (R8). p 0 with a mask and
        scale, or with an rng_dump, is another rule."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _philox_graph(probability=0.0, dump=None)
        inputs = _dropout_inputs(graph, seed=35)

        out = provider.compute_reference(graph, inputs)

        ones = np.ones(_DUMP_DIMS, dtype=np.float32)
        o, lse = dropout_reference(inputs[_Q], inputs[_K], inputs[_V], ones, 1.0)
        np.testing.assert_allclose(out[_O].data, o, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out[_STATS].data, lse, rtol=1e-5, atol=1e-5)
        explicit = _dropout_graph(probability=0.0)
        with pytest.raises(UnsupportedGraphError):
            provider.compute_reference(explicit, _dropout_inputs(explicit, seed=35))
        philox = _philox_graph(probability=0.0)
        with pytest.raises(UnsupportedGraphError, match="rng_dump"):
            provider.compute_reference(philox, _dropout_inputs(philox, seed=35))

    @pytest.mark.parametrize("absent", [False, True], ids=["null", "absent"])
    def test_sdpa_null_probability_with_seed_declines(self, absent) -> None:
        """A null or absent probability is not 0 (R0): seed and offset need a
        probability, so the graph is declined, not run without dropout."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = _philox_graph(probability=None, dump=None)
        if absent:
            del graph["nodes"][0]["attributes"]["dropout_probability"]

        with pytest.raises(UnsupportedGraphError):
            provider.compute_reference(graph, _dropout_inputs(graph, seed=36))


class TestHostMemoryEstimate:
    """The validation gate trusts this estimate, so it must count every
    full-size temporary of the CPU reference and no more: a forward SDPA
    without sinks runs on the fused kernel and holds no score matrix."""

    @staticmethod
    def _sdpa(q_dims, k_dims, attributes, page_table_dims=None, sink=False):
        def tensor(uid, dims, data_type="float"):
            return {"uid": uid, "dims": dims, "data_type": data_type}

        tensors = [
            tensor(1, q_dims),
            tensor(2, k_dims),
            tensor(3, k_dims),
            tensor(4, q_dims),
        ]
        inputs = {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3}
        if sink:
            tensors.append(tensor(8, [q_dims[1]]))
            inputs["sink_token_tensor_uid"] = 8
        if page_table_dims is not None:
            tensors += [
                tensor(5, page_table_dims, "int32"),
                tensor(6, page_table_dims, "int32"),
                tensor(7, [page_table_dims[0], 1, 1, 1], "int32"),
            ]
            inputs.update(
                page_table_k_tensor_uid=5,
                page_table_v_tensor_uid=6,
                seq_len_kv_tensor_uid=7,
            )
        return {
            "tensors": tensors,
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": inputs,
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": attributes,
                }
            ],
        }

    @pytest.mark.parametrize(
        "attributes",
        [
            {"causal_mask_bottom_right": True},
            {"left_bound": 255, "right_bound": 0},
        ],
    )
    def test_band_masks_count_the_full_mask(self, attributes) -> None:
        """Bottom-right and windowed masks are built at [q_len, kv_len] in
        float32 (twice) before SDPA runs; causal and unmasked graphs are not.
        Bottom-right with q_len == kv_len is native causal and builds none."""
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            estimate_reference_host_bytes as estimate,
        )

        q, k = [1, 2, 1024, 16], [1, 2, 512, 16]
        unmasked = estimate(self._sdpa(q, k, {}))
        assert estimate(self._sdpa(q, k, {"causal_mask": True})) == unmasked
        masked = estimate(self._sdpa(q, k, attributes))
        assert masked - unmasked >= 1024 * 512 * 4 * 2
        square = [1, 2, 1024, 16]
        native = {"causal_mask_bottom_right": True}
        assert estimate(self._sdpa(q, square, native)) == estimate(
            self._sdpa(q, square, {})
        )

    def test_sink_forward_counts_score_blocks_and_the_float32_mask(self) -> None:
        """A forward with sinks forms scores by hand and builds a float32 mask
        for its bounds; the same forward without sinks holds neither."""
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            estimate_reference_host_bytes as estimate,
        )

        dims = [1, 2, 1024, 16]
        causal = {"causal_mask": True}
        plain = estimate(self._sdpa(dims, dims, causal))
        with_sink = estimate(self._sdpa(dims, dims, causal, sink=True))
        # Three score-sized buffers (2 rows * 1024 * 1024 * 4 B each) plus the
        # 1024 * 1024 float32 mask; the sink tensor itself is negligible.
        assert with_sink - plain >= 3 * 2 * 1024 * 1024 * 4 + 1024 * 1024 * 4

    @pytest.mark.parametrize("form", ["explicit_mask", "philox"])
    def test_estimate_counts_dropout_scratch(self, form) -> None:
        """A dropout forward forms scores by hand, one query block at a time,
        and builds a float32 mask for its bounds. The same graph with the
        dropout keys removed, its tensors still listed, holds neither."""
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            estimate_reference_host_bytes as estimate,
        )

        dims, scores = [1, 2, 1024, 16], [1, 2, 1024, 1024]

        def graph(dropout: bool) -> dict:
            graph_json = self._sdpa(dims, dims, {"causal_mask": True})
            node = graph_json["nodes"][0]
            if form == "explicit_mask":
                tensors = {"dropout_mask_tensor_uid": (9, scores)}
                tensors["dropout_scale_tensor_uid"] = (10, [1])
            else:
                tensors = {"rng_dump_tensor_uid": (11, scores)}
                tensors["seed_tensor_uid"] = (12, [1])
                tensors["offset_tensor_uid"] = (13, [1])
                if dropout:
                    node["attributes"]["dropout_probability"] = 0.3
            for key, (uid, tensor_dims) in tensors.items():
                graph_json["tensors"].append(
                    {"uid": uid, "dims": tensor_dims, "data_type": "float"}
                )
                if dropout:
                    section = "outputs" if key == "rng_dump_tensor_uid" else "inputs"
                    node[section][key] = uid
            return graph_json

        # Three score blocks of 2 rows * 1024 * 1024 * 4 B (one block holds
        # every query row here) plus the 1024 * 1024 float32 causal mask.
        assert estimate(graph(True)) - estimate(graph(False)) >= (
            3 * 2 * 1024 * 1024 * 4 + 1024 * 1024 * 4
        )

    def test_paged_counts_the_kv_length_the_page_table_addresses(self) -> None:
        """K and V of a paged graph hold pages, not sequences: the gathered KV
        length is pages per sequence times the page size."""
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            estimate_reference_host_bytes as estimate,
        )

        q, pages = [4, 2, 64, 16], [64, 2, 16, 16]  # 64 pages of 16 rows
        one_page = estimate(self._sdpa(q, pages, {}, page_table_dims=[4, 1], sink=True))
        all_pages = estimate(
            self._sdpa(q, pages, {}, page_table_dims=[4, 16], sink=True)
        )
        # One block of scores for every (batch, head) row: 4 * 2 * 64 queries
        # against 16 pages * 16 rows instead of 1 page.
        assert all_pages - one_page >= 4 * 2 * 64 * (16 - 1) * 16 * 4

    def test_alibi_forward_counts_the_full_bias_tensor(self) -> None:
        """A plain ALiBi forward (no sink) still calls the real fused SDPA
        kernel once, but that call takes one [heads, q_len, kv_len] mask
        tensor in the graph dtype (float32 here)."""
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            estimate_reference_host_bytes as estimate,
        )

        dims = [1, 2, 1024, 16]
        causal = {"causal_mask": True}
        plain = estimate(self._sdpa(dims, dims, causal))
        with_alibi = estimate(self._sdpa(dims, dims, {**causal, "alibi_mask": True}))
        assert with_alibi - plain >= 2 * 1024 * 1024 * 4

    def test_alibi_bf16_forward_counts_the_stats_score_blocks(self) -> None:
        """Below float32 the kernel reads a rounded bias, so the reference
        computes the stats output again from score blocks. The same graph
        without stats holds no score block."""
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            estimate_reference_host_bytes as estimate,
        )

        def graph(stats: bool) -> dict:
            dims = [1, 2, 1024, 16]
            graph_json = self._sdpa(
                dims, dims, {"causal_mask": True, "alibi_mask": True}
            )
            for tensor in graph_json["tensors"]:
                tensor["data_type"] = "bfloat16"
            if stats:
                graph_json["tensors"].append(
                    {"uid": 9, "dims": [1, 2, 1024, 1], "data_type": "float"}
                )
                graph_json["nodes"][0]["outputs"]["stats_tensor_uid"] = 9
            return graph_json

        # Three score blocks of 2 rows * 1024 * 1024 * 4 B.
        assert estimate(graph(True)) - estimate(graph(False)) >= 3 * 2 * 1024 * 1024 * 4

    def test_alibi_backward_adds_one_block_temporary_not_the_full_bias(self) -> None:
        """Backward recomputes the ALiBi bias one query block at a time
        inside ``_block_scores``, so it must add only the block's float64 and
        float32 bias to the existing three temporaries, never a full
        ``[heads, q_len, kv_len]`` tensor (32 GiB at these dims)."""
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            estimate_reference_host_bytes as estimate,
        )

        def backward_graph(attributes):
            dims = [1, 32, 16384, 16384 // 1024]
            tensors = [
                {"uid": uid, "dims": dims, "data_type": "float"}
                for uid in (1, 2, 3, 4, 5, 7, 8, 9)
            ]
            tensors.append({"uid": 6, "dims": [1, 32, 16384, 1], "data_type": "float"})
            return {
                "tensors": tensors,
                "nodes": [
                    {
                        "type": "SdpaBackwardAttributes",
                        "inputs": {
                            "q_tensor_uid": 1,
                            "k_tensor_uid": 2,
                            "v_tensor_uid": 3,
                            "o_tensor_uid": 4,
                            "do_tensor_uid": 5,
                            "stats_tensor_uid": 6,
                        },
                        "outputs": {
                            "dq_tensor_uid": 7,
                            "dk_tensor_uid": 8,
                            "dv_tensor_uid": 9,
                        },
                        "attributes": attributes,
                    }
                ],
            }

        causal = {"causal_mask": True}
        plain = estimate(backward_graph(causal))
        with_alibi = estimate(backward_graph({**causal, "alibi_mask": True}))
        full_bias = 32 * 16384 * 16384 * 4
        assert 0 < with_alibi - plain < full_bias
        assert with_alibi - plain <= 3 * (1 << 31)  # bounded by the block cap

    def test_alibi_backward_scratch_has_no_batch_dimension(self) -> None:
        """The per-block ALiBi bias broadcasts across batch
        (``_alibi_bias_block`` builds no batch dimension), unlike the
        batch*heads-sized score temporaries it is added to. The ALiBi
        contribution to the estimate must not scale with batch, even though
        the surrounding block and expanded terms do."""
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            estimate_reference_host_bytes as estimate,
        )

        def backward_graph(batch, attributes):
            dims = [batch, 2, 1024, 16]
            tensors = [
                {"uid": uid, "dims": dims, "data_type": "float"}
                for uid in (1, 2, 3, 4, 5, 7, 8, 9)
            ]
            tensors.append(
                {"uid": 6, "dims": [batch, 2, 1024, 1], "data_type": "float"}
            )
            return {
                "tensors": tensors,
                "nodes": [
                    {
                        "type": "SdpaBackwardAttributes",
                        "inputs": {
                            "q_tensor_uid": 1,
                            "k_tensor_uid": 2,
                            "v_tensor_uid": 3,
                            "o_tensor_uid": 4,
                            "do_tensor_uid": 5,
                            "stats_tensor_uid": 6,
                        },
                        "outputs": {
                            "dq_tensor_uid": 7,
                            "dk_tensor_uid": 8,
                            "dv_tensor_uid": 9,
                        },
                        "attributes": attributes,
                    }
                ],
            }

        causal = {"causal_mask": True}
        alibi = {**causal, "alibi_mask": True}
        plain_b1 = estimate(backward_graph(1, causal))
        plain_b4 = estimate(backward_graph(4, causal))
        delta_b1 = estimate(backward_graph(1, alibi)) - plain_b1
        delta_b4 = estimate(backward_graph(4, alibi)) - plain_b4

        assert delta_b1 == delta_b4
        # Sanity: batch does scale the surrounding per-block temporaries, so
        # this is not two deltas of zero agreeing by coincidence.
        assert delta_b1 > 0
        assert plain_b4 > plain_b1


class TestFloat64Reference:
    """The float64 reference of convolution graphs."""

    PARAMS = {
        "conv_mode": "CROSS_CORRELATION",
        "pre_padding": [1, 1],
        "post_padding": [1, 1],
        "stride": [1, 1],
        "dilation": [1, 1],
    }
    X, W, DY = (2, 3, 6, 6), (4, 3, 3, 3), (2, 4, 6, 6)

    @classmethod
    def _graph(cls, node_type: str, nodes=()) -> dict:
        x, w, dy, out = {"uid": 1}, {"uid": 2}, {"uid": 3}, {"uid": 4}
        node = {"type": node_type, "parameters": cls.PARAMS}
        if node_type == "ConvolutionFwdAttributes":
            node["inputs"] = {"x_tensor_uid": 1, "w_tensor_uid": 2}
            node["outputs"] = {"y_tensor_uid": 4}
            x["dims"], w["dims"], out["dims"] = cls.X, cls.W, cls.DY
            tensors = [x, w, out]
        elif node_type == "ConvolutionBwdAttributes":
            node["inputs"] = {"dy_tensor_uid": 3, "w_tensor_uid": 2}
            node["outputs"] = {"dx_tensor_uid": 4}
            dy["dims"], w["dims"], out["dims"] = cls.DY, cls.W, cls.X
            tensors = [dy, w, out]
        else:
            node["inputs"] = {"x_tensor_uid": 1, "dy_tensor_uid": 3}
            node["outputs"] = {"dw_tensor_uid": 4}
            x["dims"], dy["dims"], out["dims"] = cls.X, cls.DY, cls.W
            tensors = [x, dy, out]
        for tensor in tensors:
            tensor["data_type"] = "half"
        return {"tensors": tensors, "nodes": [node, *nodes]}

    @staticmethod
    def _inputs() -> dict:
        rng = np.random.default_rng(0)
        shapes = {1: TestFloat64Reference.X, 2: TestFloat64Reference.W}
        shapes[3] = TestFloat64Reference.DY
        return {
            uid: rng.uniform(-1, 1, shape).astype(np.float32)
            for uid, shape in shapes.items()
        }

    @pytest.mark.parametrize(
        "node_type",
        [
            "ConvolutionFwdAttributes",
            "ConvolutionBwdAttributes",
            "ConvolutionWrwAttributes",
        ],
    )
    def test_float64_reference_is_exact_math_on_the_graph_dtype_inputs(
        self, node_type: str
    ) -> None:
        """Inputs round to fp16 as the engine sees them; the result stays
        float64 and is not rounded to fp16."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        inputs = self._inputs()
        # The fp16 values the engine sees, exactly, in float64.
        x, w, dy = (torch.from_numpy(inputs[uid]).half().double() for uid in (1, 2, 3))
        kw = {"padding": 1}
        expected = {
            "ConvolutionFwdAttributes": lambda: torch.nn.functional.conv2d(x, w, **kw),
            "ConvolutionBwdAttributes": lambda: torch.nn.grad.conv2d_input(
                self.X, w, dy, **kw
            ),
            "ConvolutionWrwAttributes": lambda: torch.nn.grad.conv2d_weight(
                x, self.W, dy, **kw
            ),
        }[node_type]().numpy()

        (out,) = provider.compute_reference(
            self._graph(node_type), inputs, precision=ReferencePrecision.FLOAT64
        ).values()

        assert out.precision == "float64"
        assert out.data.dtype == np.float64
        np.testing.assert_allclose(out.data, expected, rtol=1e-12, atol=1e-12)
        assert np.any(out.data != out.data.astype(np.float16))

    def test_native_reference_keeps_the_graph_dtype(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        (out,) = provider.compute_reference(
            self._graph("ConvolutionFwdAttributes"), self._inputs()
        ).values()

        assert (out.precision, out.data.dtype) == ("native", np.float16)

    def test_fused_pointwise_runs_in_float64(self) -> None:
        relu = {
            "type": "PointwiseAttributes",
            "inputs": {"operation": "relu_fwd", "in_0_tensor_uid": 4},
            "outputs": {"out_0_tensor_uid": 5},
        }
        graph = self._graph("ConvolutionFwdAttributes", [relu])
        graph["tensors"][-1]["virtual"] = True
        graph["tensors"].append({"uid": 5, "dims": self.DY, "data_type": "half"})
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        inputs = self._inputs()
        x, w = (torch.from_numpy(inputs[uid]).half().double() for uid in (1, 2))

        outputs = provider.compute_reference(
            graph, inputs, precision=ReferencePrecision.FLOAT64
        )

        assert list(outputs) == [5]
        np.testing.assert_allclose(
            outputs[5].data,
            torch.relu(torch.nn.functional.conv2d(x, w, padding=1)).numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "types, supported",
        [
            (["ConvolutionWrwAttributes"], True),
            (["ConvolutionFwdAttributes", "PointwiseAttributes"], True),
            (["PointwiseAttributes"], False),
            (["ConvolutionFwdAttributes", "BatchnormAttributes"], False),
            (["MatmulAttributes"], False),
        ],
    )
    def test_float64_covers_convolution_and_pointwise_graphs_only(
        self, types: list, supported: bool
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = {"nodes": [{"type": t} for t in types]}

        assert provider.supports_float64(graph) is supported

    def test_float64_request_for_an_unsupported_graph_raises(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph = {
            "tensors": [{"uid": 1, "dims": [2]}, {"uid": 2, "dims": [2]}],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {"operation": "relu_fwd", "in_0_tensor_uid": 1},
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ],
        }

        with pytest.raises(UnsupportedGraphError, match="float64"):
            provider.compute_reference(
                graph,
                {1: np.ones(2, dtype=np.float32)},
                precision=ReferencePrecision.FLOAT64,
            )
