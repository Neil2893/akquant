"""Version sanity check tests."""

from typing import Any

import akquant as aq


def test_version_present() -> None:
    """
    Ensure package exposes a non-empty __version__ string.

    :return: None
    """
    v: Any = aq.__version__
    assert isinstance(v, str)
    assert len(v) > 0
