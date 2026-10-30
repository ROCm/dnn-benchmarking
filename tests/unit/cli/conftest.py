# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import pytest

from dnn_benchmarking.cli.main import _WORKSPACE_CACHE_SUBDIRS
from dnn_benchmarking.common.run_state import TEMP_ROOT_ENV


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
        TEMP_ROOT_ENV,
    ):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
