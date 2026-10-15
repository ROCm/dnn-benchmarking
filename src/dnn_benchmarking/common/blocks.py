# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Split large arrays into blocks to bound full-size temporaries.

Dtype conversions and comparisons make several temporaries the size of their
operand. On a multi-GiB tensor these temporaries, not the data, exhaust host
memory. Processing one block at a time bounds them to a block.
"""

import math
from typing import Iterator, Optional, Tuple, Union

#: Elements in one block. The bfloat16 and fp8 conversions need up to about
#: 20 bytes per element of temporaries, so one block uses tens of MiB.
BLOCK_ELEMENTS = 1 << 22

Index = Tuple[Union[int, slice], ...]


def iter_blocks(shape: Tuple[int, ...], limit: Optional[int] = None) -> Iterator[Index]:
    """Yield index tuples that split an array of ``shape`` into blocks.

    Each block has at most ``limit`` elements (default ``BLOCK_ELEMENTS``,
    at least 1). The blocks cover the array once, in C order. An index holds
    integers for the leading dimensions it fixes, then at most one slice;
    the remaining dimensions are whole.
    """
    if limit is None:
        limit = BLOCK_ELEMENTS
    if not shape or math.prod(shape) <= limit:
        yield ()
        return
    inner = math.prod(shape[1:])
    if inner <= limit:
        step = limit // inner
        for start in range(0, shape[0], step):
            yield (slice(start, min(start + step, shape[0])),)
        return
    for i in range(shape[0]):
        for rest in iter_blocks(shape[1:], limit):
            yield (i,) + rest


def global_index(
    index: Index, block_shape: Tuple[int, ...], flat: int
) -> Tuple[int, ...]:
    """Array coordinates of element ``flat`` (C order) of the block at ``index``."""
    coords = []
    for position in range(len(block_shape)):
        coords.append(flat % block_shape[-1 - position])
        flat //= block_shape[-1 - position]
    local = iter(reversed(coords))
    result = []
    for item in index:
        if isinstance(item, slice):
            result.append(item.start + next(local))
        else:
            result.append(item)
    result.extend(local)
    return tuple(int(i) for i in result)
