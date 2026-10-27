# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import os

import pytest

from dnn_benchmarking.cli.main import _CACHE_ENV_SUBDIRS
from dnn_benchmarking.common import selection_env


@pytest.fixture(autouse=True)
def _restore_cli_env(monkeypatch):
    """Start each test with the variables the CLI reads or writes unset.

    A selection variable inherited from the developer's or CI's shell would
    make every run exit 2. delenv alone records nothing for an absent
    variable, so a direct os.environ write would outlive the test; setenv
    first records it.
    """
    names = {*selection_env.SELECTION_ENV, *_CACHE_ENV_SUBDIRS}
    names |= set(selection_env.inherited(os.environ))
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
