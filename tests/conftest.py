from __future__ import annotations

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate",
        action="store_true",
        default=False,
        help="Regenerate reference values for tests that support it.",
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CAPM_SKIP_CUDA_TESTS") != "1":
        return

    skip_cuda = pytest.mark.skip(reason="Skipped in CPU target (CAPM_SKIP_CUDA_TESTS=1)")
    cuda_tokens = ("cuda", "gpu")
    for item in items:
        nodeid = item.nodeid.lower()
        if any(token in nodeid for token in cuda_tokens):
            item.add_marker(skip_cuda)
