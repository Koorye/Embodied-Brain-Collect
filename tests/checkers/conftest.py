"""Fixtures for the checker tests."""

from pathlib import Path

import numpy as np
import pytest

from embodied_brain_collect.checkers.base import CheckContext

REPO = Path(__file__).resolve().parents[2]
SESSION4 = REPO / "data" / "session4"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: needs the real session fixtures (decodes video)")


@pytest.fixture
def ctx(tmp_path):
    """A context wired to synthetic data instead of a recorder directory.

    ``feed(series=..., arrays=...)`` registers timestamp series and NPZ
    fields without touching disk, so a check can be exercised on exactly the
    shape it is meant to catch.
    """

    def make(*, series=None, arrays=None, window=None, default=None):
        c = CheckContext(stream="test", directory=tmp_path, window=window,
                         default_series=default or (
                             next(iter(series), None) if series else None))
        for label, values in (series or {}).items():
            c.add_series(label, loader=(lambda v=values: np.asarray(v, float)))
        for key, value in (arrays or {}).items():
            c._arrays[key] = np.asarray(value)
        return c

    return make


@pytest.fixture(scope="session")
def session4():
    if not SESSION4.is_dir():
        pytest.skip(f"missing fixture {SESSION4}")
    return SESSION4
