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


def sdpa_graph(b=1, hq=32, hkv=8, s=2048, d=128, dtype="half", **attributes):
    """An SDPA forward graph laid out like the rocKE perf-oracle workloads."""
    attrs = {"causal_mask": True, "attn_scale_value": None}
    attrs.update(attributes)
    return {
        "name": "sdpa",
        "io_data_type": dtype,
        "compute_data_type": "float",
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
                "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": None},
                "attributes": attrs,
            }
        ],
    }
