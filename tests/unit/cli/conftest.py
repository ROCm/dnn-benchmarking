# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import pytest

from dnn_benchmarking.common.work_dir import COMPILE_CACHE_SUBDIRS


@pytest.fixture(autouse=True)
def _restore_cli_env(monkeypatch, tmp_path):
    """Start each test with the variables the CLI writes to os.environ unset.

    delenv alone records nothing for an absent variable, so a direct
    os.environ write would outlive the test; setenv first records it.
    """
    names = (
        "HIPDNN_FORCE_BENCHMARKING",
        "HIPDNN_CACHE_DIR",
        "MIOPEN_USER_DB_PATH",
        *COMPILE_CACHE_SUBDIRS,
    )
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    # The default work dir: never the user's real one.
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path / "workspace"))
