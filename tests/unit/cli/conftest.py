# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import pytest

from dnn_benchmarking.cli.main import _MIOPEN_USER_DB, _WORKSPACE_CACHE_SUBDIRS


@pytest.fixture(autouse=True)
def _restore_cli_env(monkeypatch):
    """Start each test with the variables the CLI writes to os.environ unset.

    delenv alone records nothing for an absent variable, so a direct
    os.environ write would outlive the test; setenv first records it.
    """
    for name in (
        "HIPDNN_FORCE_BENCHMARKING",
        "HIPDNN_CACHE_DIR",
        "HIPDNN_DISABLE_CACHE",
        *_WORKSPACE_CACHE_SUBDIRS,
        _MIOPEN_USER_DB,
    ):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
