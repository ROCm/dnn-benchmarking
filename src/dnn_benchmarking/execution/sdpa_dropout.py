# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Read the dropout attributes of SDPA nodes without importing torch.

The suite runner, the tuned child and the buffer manager use this module, so
it must stay importable on the hipDNN path. The PyTorch handler reads dropout
through the same functions.
"""

from typing import Any, Dict, Iterator, NamedTuple, Optional, Tuple


class SdpaDropout(NamedTuple):
    """The dropout attributes of one SDPA node.

    A field is None when its key is absent or JSON null. A probability of 0.0
    stays 0.0: hipDNN reads a non-null 0 as "no dropout", and a null as "not
    given".
    """

    probability: Optional[float]
    seed_uid: Optional[int]
    offset_uid: Optional[int]
    mask_uid: Optional[int]  # dropout_mask_tensor_uid
    scale_uid: Optional[int]  # dropout_scale_tensor_uid
    scale_inv_uid: Optional[int]  # dropout_scale_inv_tensor_uid (backward nodes)
    rng_dump_uid: Optional[int]  # rng_dump_tensor_uid (forward outputs)


class RngDump(NamedTuple):
    """The ``rng_dump`` output of a Philox dropout forward node."""

    uid: int
    probability: float


_FORWARD = "SdpaAttributes"

# The lookup order of pytorch_ops/_common.py (_node_param, _optional_uid). That
# module imports torch, so the order is repeated here. A null value counts as
# absent, and the search goes on to the next place.
_PARAM_SECTIONS = ("parameters", "attributes", "inputs", "outputs")
_UID_SECTIONS = ("inputs", "outputs", "attributes")


def _first_non_null(node: Dict[str, Any], key: str, sections: Tuple[str, ...]) -> Any:
    for name in sections:
        section = node.get(name)
        if isinstance(section, dict) and section.get(key) is not None:
            return section[key]
    return node.get(key)


def _uid(node: Dict[str, Any], key: str) -> Optional[int]:
    value = _first_non_null(node, key, _UID_SECTIONS)
    return None if value is None else int(value)


def read_dropout(node: Dict[str, Any]) -> SdpaDropout:
    """Read the dropout attributes of one SDPA node."""
    probability = _first_non_null(node, "dropout_probability", _PARAM_SECTIONS)
    return SdpaDropout(
        probability=None if probability is None else float(probability),
        seed_uid=_uid(node, "seed_tensor_uid"),
        offset_uid=_uid(node, "offset_tensor_uid"),
        mask_uid=_uid(node, "dropout_mask_tensor_uid"),
        scale_uid=_uid(node, "dropout_scale_tensor_uid"),
        scale_inv_uid=_uid(node, "dropout_scale_inv_tensor_uid"),
        rng_dump_uid=_uid(node, "rng_dump_tensor_uid"),
    )


def _forward_dropouts(
    graph_json: Dict[str, Any],
) -> Iterator[Tuple[str, SdpaDropout]]:
    for node in graph_json.get("nodes", []):
        if node.get("type") == _FORWARD:
            yield str(node.get("name") or _FORWARD), read_dropout(node)


def find_rng_dump(graph_json: Dict[str, Any]) -> Optional[RngDump]:
    """The first forward SDPA node with an ``rng_dump`` and a probability.

    hipDNN writes the dump only when the probability is not null
    (RngDumpHandoff.hpp, rocm-libraries #14358; the submodule pin predates
    it). Backward nodes have no dump.
    """
    for _, dropout in _forward_dropouts(graph_json):
        if dropout.rng_dump_uid is not None and dropout.probability is not None:
            return RngDump(dropout.rng_dump_uid, dropout.probability)
    return None


def philox_without_dump(graph_json: Dict[str, Any]) -> Optional[str]:
    """A message when a forward SDPA node has Philox dropout and no ``rng_dump``.

    Philox dropout is a probability above 0 with no explicit mask or scale.
    The PyTorch reference cannot make the Philox random stream again, so it
    needs the mask that the engine writes to ``rng_dump``.
    """
    for name, dropout in _forward_dropouts(graph_json):
        if (
            dropout.probability is not None
            and dropout.probability > 0.0
            and dropout.rng_dump_uid is None
            and dropout.mask_uid is None
            and dropout.scale_uid is None
        ):
            return (
                f"SDPA node '{name}' has Philox dropout "
                f"(dropout_probability={dropout.probability}) and no rng_dump "
                "output. The PyTorch reference cannot make the Philox random "
                "stream again. Add an rng_dump output to validate this graph."
            )
    return None
