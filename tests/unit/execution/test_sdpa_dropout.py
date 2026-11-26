# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.sdpa_dropout (dropout attributes of SDPA nodes)."""

import pytest

from dnn_benchmarking.execution.sdpa_dropout import (
    RngDump,
    find_rng_dump,
    philox_without_dump,
    read_dropout,
)

_DUMP_UID = 9


def _node(node_type="SdpaAttributes", **sections):
    node = {"name": "sdpa", "type": node_type}
    node.update(sections)
    return node


def _graph(*nodes):
    return {"nodes": list(nodes)}


def _philox(probability, **outputs):
    """A forward node with seed and offset inputs and ``probability``."""
    return _node(
        inputs={"q_tensor_uid": 1, "seed_tensor_uid": 5, "offset_tensor_uid": 6},
        outputs={"o_tensor_uid": 4, **outputs},
        attributes={"dropout_probability": probability},
    )


@pytest.mark.parametrize(
    "node, expected",
    [
        (_node(attributes={}), None),
        (_node(attributes={"dropout_probability": None}), None),
        (_node(attributes={"dropout_probability": 0.0}), 0.0),
        (_node(parameters={"dropout_probability": 0.3}), 0.3),
        (_node(attributes={"dropout_probability": 0.3}), 0.3),
        (_node(inputs={"dropout_probability": 0.3}), 0.3),
        (_node(outputs={"dropout_probability": 0.3}), 0.3),
        (_node(dropout_probability=0.3), 0.3),
        (
            _node(
                parameters={"dropout_probability": 0.2},
                attributes={"dropout_probability": None},
            ),
            0.2,
        ),
        (
            _node(
                parameters={"dropout_probability": None},
                attributes={"dropout_probability": 0.2},
            ),
            0.2,
        ),
    ],
    ids=[
        "absent",
        "null",
        "zero",
        "parameters",
        "attributes",
        "inputs",
        "outputs",
        "top_level",
        "parameters_over_null_attributes",
        "null_parameters_does_not_hide_attributes",
    ],
)
def test_null_or_absent_probability_is_not_present(node, expected):
    probability = read_dropout(node).probability

    if expected is None:
        assert probability is None
    else:
        assert probability == expected


@pytest.mark.parametrize(
    "graph, expected",
    [
        (_graph(_philox(None, rng_dump_tensor_uid=_DUMP_UID)), None),
        (_graph(_philox(0.3, rng_dump_tensor_uid=_DUMP_UID)), RngDump(_DUMP_UID, 0.3)),
        (
            _graph(
                _node(
                    inputs={"seed_tensor_uid": 5, "offset_tensor_uid": 6},
                    outputs={"rng_dump_tensor_uid": _DUMP_UID},
                    parameters={"dropout_probability": 0.3},
                )
            ),
            RngDump(_DUMP_UID, 0.3),
        ),
        (
            _graph(
                _node(
                    "SdpaBackwardAttributes",
                    inputs={"seed_tensor_uid": 5, "offset_tensor_uid": 6},
                    outputs={"rng_dump_tensor_uid": _DUMP_UID},
                    parameters={"dropout_probability": 0.3},
                )
            ),
            None,
        ),
    ],
    ids=["null_probability", "probability", "probability_in_parameters", "backward"],
)
def test_find_rng_dump_needs_a_non_null_probability(graph, expected):
    assert find_rng_dump(graph) == expected


@pytest.mark.parametrize(
    "node, missing_dump",
    [
        (_philox(0.3), True),
        (_philox(None), False),
        (_philox(0.0), False),
        (_philox(0.3, rng_dump_tensor_uid=_DUMP_UID), False),
        # A probability with a mask and scale is an explicit-mask graph that
        # the handler declines for its own reason, not a missing dump.
        (
            _node(
                inputs={
                    "q_tensor_uid": 1,
                    "dropout_mask_tensor_uid": 7,
                    "dropout_scale_tensor_uid": 8,
                },
                outputs={"o_tensor_uid": 4},
                attributes={"dropout_probability": 0.3},
            ),
            False,
        ),
    ],
    ids=["no_dump", "null_probability", "zero_probability", "dump", "mask_and_scale"],
)
def test_philox_without_dump_names_rng_dump(node, missing_dump):
    message = philox_without_dump(_graph(node))

    if missing_dump:
        assert message is not None and "rng_dump" in message
    else:
        assert message is None
