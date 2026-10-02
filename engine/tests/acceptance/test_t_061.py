"""Acceptance tests for T-061: the review window is always destroyed on teardown, even
when cancelling a scheduled Tcl callback fails. The task contract: do not edit.

Every image here is synthetic. The tests need Tk and a display: they skip without one,
except on Windows or with WEARREPORT_REQUIRE_TK set (on Linux, run them under
`xvfb-run`). Nothing is written to disk.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pytest

from wearreport.tools import spotcheck

WAIT_S = 60
REQUIRE_TK = "WEARREPORT_REQUIRE_TK"


def _need_window() -> None:
    try:
        import tkinter

        tkinter.Tk().destroy()
    except Exception as exc:  # ImportError without Tk, TclError without a display
        if sys.platform == "win32" or os.environ.get(REQUIRE_TK):
            pytest.fail(f"no Tk window here: {exc}")
        pytest.skip("no Tk display here (on Linux run under xvfb-run)")


def _crops(numbers: Sequence[int]) -> list[spotcheck.ReviewItem]:
    image = np.full((60, 40, 3), 90, dtype=np.uint8)
    return [spotcheck.ReviewItem(n, f"crop-{n:04d}.png", (n,), image) for n in numbers]


def _closed(root: Any) -> bool:
    import tkinter

    try:
        return not root.winfo_exists()
    except tkinter.TclError:
        return True


@pytest.fixture
def cancel_fails(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """The first `after_cancel` raises TclError, as tkinter's does for an id whose
    callback already ran; later ones cancel as usual. Yields the ids it was asked for."""
    _need_window()
    import tkinter

    real = tkinter.Misc.after_cancel
    asked: list[str] = []

    def after_cancel(self: Any, id: str) -> None:
        asked.append(id)
        if len(asked) == 1:
            raise tkinter.TclError(f'event "{id}" doesn\'t exist')
        real(self, id)

    monkeypatch.setattr(tkinter.Misc, "after_cancel", after_cancel)
    yield asked


class Keys:
    """A window driver: keeps two long callbacks scheduled (so teardown has several to
    cancel), then sends `keys` 20 ms apart."""

    def __init__(self, *keys: str) -> None:
        self.keys = list(keys)
        self.root: Any = None

    def __call__(self, root: Any) -> None:
        self.root = root
        root.after(WAIT_S * 10_000, int)
        root.after(WAIT_S * 10_000, int)
        root.after(20, self._step)

    def _step(self) -> None:
        if self.keys:
            self.root.focus_force()
            self.root.event_generate(self.keys.pop(0))
            self.root.after(20, self._step)


def test_ac2_judge_timeout_destroys_the_window_when_a_cancel_fails(
    cancel_fails: list[str],
) -> None:
    keys = Keys()  # never answers
    with pytest.raises(spotcheck.ReviewTimeout):
        spotcheck.WindowReviewer(driver=keys).judge(_crops([1]), "crops", time.monotonic() + 0.5)
    assert cancel_fails, "teardown cancelled nothing: the test did not exercise the failure"
    assert _closed(keys.root)


def test_ac2_judge_normal_path_destroys_the_window_when_a_cancel_fails(
    cancel_fails: list[str],
) -> None:
    keys = Keys("<Return>", "<KeyPress-n>")
    reviewer = spotcheck.WindowReviewer(driver=keys)
    got = reviewer.judge(_crops([1, 2]), "crops", time.monotonic() + WAIT_S)
    assert sorted(got) == [1, 2]
    assert cancel_fails
    assert _closed(keys.root)


def test_ac2_attributes_timeout_destroys_the_window_when_a_cancel_fails(
    cancel_fails: list[str],
) -> None:
    keys = Keys()
    with pytest.raises(spotcheck.ReviewTimeout):
        spotcheck.WindowReviewer(driver=keys).attributes(_crops([1]), time.monotonic() + 0.5)
    assert cancel_fails
    assert _closed(keys.root)


def test_ac2_attributes_normal_path_destroys_the_window_when_a_cancel_fails(
    cancel_fails: list[str],
) -> None:
    keys = Keys("<KeyPress-y>", "<KeyPress-n>", "<KeyPress-u>")
    reviewer = spotcheck.WindowReviewer(driver=keys)
    got = reviewer.attributes(_crops([1]), time.monotonic() + WAIT_S)
    assert dict(got) == {1: "ynu"}
    assert cancel_fails
    assert _closed(keys.root)
