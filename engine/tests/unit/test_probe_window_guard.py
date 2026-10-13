"""Deterministic injected-window teardown tests through the real collection paths.

The Tk/window doubles expose teardown milestones without a display or wall-clock
race. WindowReviewer and the active SignalGuard execute their production methods.
"""

from __future__ import annotations

import datetime
import functools
import signal
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport.tools import spotcheck


class _Root:
    def __init__(self) -> None:
        self.tk = self
        self.destroyed = False
        self.cancelled: list[str] = []
        self.report_callback_exception: object = None

    def call(self, *args: str) -> tuple[str, ...]:
        assert args == ("after", "info")
        return ("tick", "driver")

    def splitlist(self, pending: tuple[str, ...]) -> tuple[str, ...]:
        return pending

    def after_cancel(self, pending: str) -> None:
        self.cancelled.append(pending)

    def destroy(self) -> None:
        self.destroyed = True


class _Windows:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.roots: list[_Root] = []
        self.closed = 0
        self.during_close: Callable[[], None] = lambda: None
        self.run_failure: BaseException | None = None
        tk = ModuleType("synthetic_tk")

        def root() -> _Root:
            result = _Root()
            self.roots.append(result)
            return result

        tk.__dict__.update(Tk=root, TclError=RuntimeError)
        monkeypatch.setattr(spotcheck, "_import_tkinter", lambda: tk)
        owner = self

        class Window:
            def __init__(
                self, tk: Any, root: Any, items: Any, deadline: float, confirm_stop: bool
            ) -> None:
                self.root, self.items, self.deadline = root, items, deadline
                self.fit_screen = False

            def run(self, driver: Callable[[Any], object] | None) -> dict[int, Any]:
                if driver is not None:
                    driver(self.root)
                if owner.run_failure is not None:
                    raise owner.run_failure
                return {item.number: self.answer() for item in self.items}

            def answer(self) -> object:
                return spotcheck.Judgement(frozenset(), frozenset(), None)

            def close(self) -> None:
                owner.closed += 1
                owner.during_close()

        class Attributes(Window):
            def answer(self) -> str:
                return "ynn"

        monkeypatch.setattr(spotcheck, "_ReviewWindow", Window)
        monkeypatch.setattr(spotcheck, "_AttributeWindow", Attributes)

    def assert_destroyed(self) -> None:
        assert self.roots[-1].destroyed
        assert self.roots[-1].cancelled == ["tick", "driver"]
        assert not hasattr(self.roots[-1], "report_callback_exception")


def _items() -> list[spotcheck.ReviewItem]:
    image = np.zeros((60, 40, 3), dtype=np.uint8)
    return [spotcheck.ReviewItem(1, "crop-0001.png", (1,), image)]


def _collect(
    attributes: bool, reviewer: spotcheck.WindowReviewer, guard: spotcheck._SignalGuard
) -> object:
    if attributes:
        return spotcheck._collect_attributes(_items(), reviewer, 60, guard)
    return spotcheck._collect(_items(), "crops", reviewer, 60, guard)


def _timeout(guard: spotcheck._SignalGuard) -> None:
    guard._alarm_due = True
    guard._handle(signal.SIGINT, None)


@pytest.mark.parametrize("attributes", [False, True])
def test_injected_window_defers_teardown_alarm_and_rebinds_on_reuse(
    attributes: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows = _Windows(monkeypatch)
    reviewer = spotcheck.WindowReviewer()
    observed: list[spotcheck._SignalGuard | None] = []
    reviewer.driver = lambda root: observed.append(reviewer.guard)
    for _ in range(2):
        active = spotcheck._SignalGuard()
        windows.during_close = functools.partial(_timeout, active)
        with pytest.raises(spotcheck.ReviewTimeout, match="the review timed out"):
            _collect(attributes, reviewer, active)
        windows.assert_destroyed()
        assert reviewer.guard is None
        assert active._stopping
        assert observed[-1] is active
    assert observed[0] is not observed[1]
    assert windows.closed == 2


@pytest.mark.parametrize("attributes", [False, True])
@pytest.mark.parametrize(
    "outcome", ["success", "run-timeout", "run-failure", "close-failure", "invalid-answer"]
)
def test_temporary_guard_is_restored_after_every_collection_outcome(
    attributes: bool, outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows = _Windows(monkeypatch)
    reviewer, active = spotcheck.WindowReviewer(), spotcheck._SignalGuard()
    observed: list[spotcheck._SignalGuard | None] = []
    reviewer.driver = lambda root: observed.append(reviewer.guard)
    if outcome == "run-timeout":
        windows.run_failure = spotcheck.ReviewTimeout("the review timed out")
    elif outcome == "run-failure":
        windows.run_failure = ValueError("synthetic run failure")
    elif outcome == "close-failure":

        def fail_close() -> None:
            raise ValueError("synthetic close failure")

        windows.during_close = fail_close
    elif outcome == "invalid-answer":
        method = "validate_attributes" if attributes else "validate_all"

        def invalid(*args: object) -> object:
            raise spotcheck.JudgementError("synthetic invalid answer")

        monkeypatch.setattr(spotcheck, method, invalid)
    if outcome == "success":
        assert _collect(attributes, reviewer, active)
    else:
        expected = {
            "run-timeout": spotcheck.ReviewTimeout,
            "run-failure": ValueError,
            "close-failure": ValueError,
            "invalid-answer": spotcheck.SpotcheckError,
        }[outcome]
        with pytest.raises(expected):
            _collect(attributes, reviewer, active)
    assert reviewer.guard is None
    assert observed == [active]
    if outcome != "close-failure":
        windows.assert_destroyed()


@pytest.mark.parametrize("attributes", [False, True])
def test_explicit_caller_guard_controls_teardown_and_is_preserved(
    attributes: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows = _Windows(monkeypatch)
    supplied, active = spotcheck._SignalGuard(), spotcheck._SignalGuard()
    reviewer = spotcheck.WindowReviewer(guard=supplied)
    observed: list[spotcheck._SignalGuard | None] = []
    reviewer.driver = lambda root: observed.append(reviewer.guard)
    windows.during_close = lambda: _timeout(supplied)
    with pytest.raises(spotcheck.ReviewTimeout):
        _collect(attributes, reviewer, active)
    windows.assert_destroyed()
    assert reviewer.guard is supplied
    assert observed == [supplied]
    assert supplied._stopping and not active._stopping


@pytest.mark.parametrize("attributes", [False, True])
def test_main_admits_injected_window_and_closes_before_timeout_exit(
    attributes: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows = _Windows(monkeypatch)
    for name in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    guards: list[spotcheck._SignalGuard] = []

    class ActiveGuard(spotcheck._SignalGuard):
        def __init__(self) -> None:
            super().__init__()
            guards.append(self)

    monkeypatch.setattr(spotcheck, "_SignalGuard", ActiveGuard)
    reviewer = spotcheck.WindowReviewer()
    windows.during_close = lambda: _timeout(guards[-1])

    class Detector:
        def detect(self, image: npt.NDArray[np.uint8]) -> list[detect.Detection]:
            return [detect.Detection("person", 0.95, (20.0, 20.0, 35.0, 86.0))]

    frames = [np.zeros((120, 100, 3), dtype=np.uint8)]
    info = spotcheck.DetectorInfo("synthetic", "0" * 64, detect.DEFAULT_CONF)
    argv = [
        "--n",
        "1",
        "--min-persons",
        "1",
        "--view",
        "window",
        "--out-dir",
        str(tmp_path / "out"),
    ]
    if attributes:
        argv += ["--attributes", "--allow-dark"]
    assert (
        spotcheck.main(
            argv,
            pipeline=spotcheck.Pipeline(lambda: frames, Detector(), info),
            reviewer=reviewer,
            today=datetime.date(2026, 10, 10),
            clock=lambda: datetime.datetime(2026, 10, 10, 12, tzinfo=datetime.UTC),
        )
        == 3
    )
    windows.assert_destroyed()
    assert reviewer.guard is None
    assert not (tmp_path / "out").exists()
