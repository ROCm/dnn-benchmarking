# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Read code objects from a ``.kpack`` archive with ROCm's kpack reader.

hip-kernel-provider ships its compiled kernels in one kpack archive per arch
and loads them through ``librocm_kpack``. This module calls the same C API
through ctypes, so the rocKE runtime reads the bytes that hipDNN loads.
"""

import ctypes
import os
from functools import lru_cache
from pathlib import Path

from ..common.rocm_runtime import _import_rocm_sdk

_SONAME = "librocm_kpack.so.0"

#: ``kpack_error_t`` (``rocm_kpack/kpack_types.h``).
_ERRORS = (
    "SUCCESS",
    "INVALID_ARGUMENT",
    "FILE_NOT_FOUND",
    "INVALID_FORMAT",
    "UNSUPPORTED_VERSION",
    "KERNEL_NOT_FOUND",
    "DECOMPRESSION_FAILED",
    "OUT_OF_MEMORY",
    "NOT_IMPLEMENTED",
    "IO_ERROR",
    "MSGPACK_PARSE_FAILED",
)


def _library_path() -> str:
    """``librocm_kpack``: ``$ROCM_PATH/lib``, the pip ROCm SDK, else the soname."""
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path and (Path(rocm_path) / "lib" / _SONAME).is_file():
        return str(Path(rocm_path) / "lib" / _SONAME)
    rocm_sdk = _import_rocm_sdk()
    if rocm_sdk is not None:
        try:
            return str(rocm_sdk.find_libraries("rocm_kpack")[0])
        except (ModuleNotFoundError, FileNotFoundError, IndexError):
            pass
    return _SONAME


@lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(_library_path())
    handle_p = ctypes.POINTER(ctypes.c_void_p)
    lib.kpack_open.argtypes = [ctypes.c_char_p, handle_p]
    lib.kpack_open.restype = ctypes.c_int
    lib.kpack_close.argtypes = [ctypes.c_void_p]
    lib.kpack_close.restype = None
    lib.kpack_get_kernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        handle_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.kpack_get_kernel.restype = ctypes.c_int
    lib.kpack_free_kernel.argtypes = [ctypes.c_void_p]
    lib.kpack_free_kernel.restype = None
    return lib


def _check(code: int, what: str) -> None:
    if code:
        name = _ERRORS[code] if code < len(_ERRORS) else str(code)
        raise OSError(f"{what} failed: KPACK_ERROR_{name}")


def read_kernel(archive: Path, toc_key: str, arch: str) -> bytes:
    """The code object stored under ``toc_key`` for ``arch``.

    Raises:
        OSError: The library does not load, or the archive is unreadable or
            has no such kernel.
    """
    lib = _lib()
    handle = ctypes.c_void_p()
    _check(
        lib.kpack_open(str(archive).encode(), ctypes.byref(handle)), f"open {archive}"
    )
    try:
        data = ctypes.c_void_p()
        size = ctypes.c_size_t()
        _check(
            lib.kpack_get_kernel(
                handle,
                toc_key.encode(),
                arch.encode(),
                ctypes.byref(data),
                ctypes.byref(size),
            ),
            f"read {toc_key} ({arch}) from {archive}",
        )
        try:
            return ctypes.string_at(data, size.value)
        finally:
            lib.kpack_free_kernel(data)
    finally:
        lib.kpack_close(handle)
