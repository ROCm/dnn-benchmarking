# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""kpack reader: the bytes it returns, its errors, and its cleanup."""

import ctypes
from pathlib import Path

import pytest

from dnn_benchmarking.execution import kpack

_CODE = b"\x7fELF code object"
_ARCHIVE = Path("/p/a.kpack")


class _Lib:
    """``librocm_kpack`` stand-in that records frees and closes."""

    def __init__(self, open_code=0, get_code=0):
        self.open_code, self.get_code = open_code, get_code
        self.buffer = ctypes.create_string_buffer(_CODE, len(_CODE))
        self.calls = []

    def kpack_open(self, path, handle_ref):
        self.calls.append(("open", path))
        if not self.open_code:
            handle_ref._obj.value = 1234
        return self.open_code

    def kpack_get_kernel(self, handle, toc_key, arch, data_ref, size_ref):
        self.calls.append(("get", toc_key, arch))
        if not self.get_code:
            data_ref._obj.value = ctypes.addressof(self.buffer)
            size_ref._obj.value = len(_CODE)
        return self.get_code

    def kpack_free_kernel(self, data):
        self.calls.append(("free",))

    def kpack_close(self, handle):
        self.calls.append(("close", handle.value))


@pytest.fixture
def lib(monkeypatch):
    def install(**codes):
        fake = _Lib(**codes)
        monkeypatch.setattr(kpack, "_lib", lambda: fake)
        return fake

    return install


def test_read_returns_the_code_object_and_frees_it(lib) -> None:
    fake = lib()
    assert kpack.read_kernel(_ARCHIVE, "toc", "gfx950") == _CODE
    assert fake.calls == [
        ("open", str(_ARCHIVE).encode()),
        ("get", b"toc", b"gfx950"),
        ("free",),
        ("close", 1234),
    ]


@pytest.mark.parametrize(
    "code, name",
    [(5, "KERNEL_NOT_FOUND"), (14, "ARCH_NOT_FOUND"), (99, "99")],
)
def test_a_failed_read_names_the_error_and_closes_the_archive(lib, code, name) -> None:
    fake = lib(get_code=code)
    with pytest.raises(OSError, match=f"read toc \\(gfx950\\).*KPACK_ERROR_{name}$"):
        kpack.read_kernel(_ARCHIVE, "toc", "gfx950")
    # Nothing was returned, so nothing is freed; the open archive is closed.
    assert fake.calls[-1] == ("close", 1234)
    assert ("free",) not in fake.calls


def test_a_failed_open_closes_nothing(lib) -> None:
    fake = lib(open_code=2)
    with pytest.raises(OSError, match="open .*KPACK_ERROR_FILE_NOT_FOUND"):
        kpack.read_kernel(_ARCHIVE, "toc", "gfx950")
    assert fake.calls == [("open", str(_ARCHIVE).encode())]
