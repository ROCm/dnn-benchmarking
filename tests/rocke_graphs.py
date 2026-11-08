# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""SDPA forward graphs shaped like the rocKE perf-oracle workloads."""


def bshd_tensor(uid, name, b, h, s, d, dtype):
    """A packed BSHD tensor in graph order ``[B, H, S, D]``."""
    return {
        "uid": uid,
        "name": name,
        "dims": [b, h, s, d],
        "strides": [s * h * d, d, h * d, 1],
        "data_type": dtype,
        "virtual": False,
    }


_INPUTS = (
    "attn_mask", "scale", "seq_len_q", "seq_len_kv", "seed", "offset",
    "dropout_mask", "dropout_scale", "page_table_k", "page_table_v",
    "block_mask", "sink_token", "descale_q", "descale_k", "descale_v",
    "descale_s", "scale_s", "scale_o",
)  # fmt: skip
_OUTPUTS = ("stats", "max", "sum_exp", "rng_dump", "amax_s", "amax_o")
#: hipDNN deserializes a node only with every attribute present.
_ATTRIBUTES = {
    "generate_stats": None,
    "alibi_mask": False,
    "padding_mask": False,
    "causal_mask": True,
    "causal_mask_bottom_right": False,
    "dropout_probability": None,
    "attn_scale_value": None,
    "left_bound": None,
    "right_bound": None,
    "max_seq_len_kv": None,
    "diagonal_alignment": "TOP_LEFT",
    "mma_core_mode": "UNSET",
    "implementation": "AUTO",
}


def sdpa_graph(b=1, hq=32, hkv=8, s=2048, d=128, dtype="half", **attributes):
    """An SDPA forward graph in the form of the rocKE perf-oracle workloads.

    hipDNN can deserialize it, so it also runs through hipDNN engines.
    """
    return {
        "name": "sdpa",
        "compute_data_type": "float",
        "io_data_type": dtype,
        "intermediate_data_type": dtype,
        "tensors": [
            bshd_tensor(1, "query", b, hq, s, d, dtype),
            bshd_tensor(2, "key", b, hkv, s, d, dtype),
            bshd_tensor(3, "value", b, hkv, s, d, dtype),
            bshd_tensor(4, "output", b, hq, s, d, dtype),
        ],
        "nodes": [
            {
                "name": "sdpa_node",
                "type": "SdpaAttributes",
                "compute_data_type": "float",
                "inputs": {
                    "q_tensor_uid": 1,
                    "k_tensor_uid": 2,
                    "v_tensor_uid": 3,
                    **{f"{name}_tensor_uid": None for name in _INPUTS},
                },
                "outputs": {
                    "o_tensor_uid": 4,
                    **{f"{name}_tensor_uid": None for name in _OUTPUTS},
                },
                "attributes": {**_ATTRIBUTES, **attributes},
            }
        ],
    }
