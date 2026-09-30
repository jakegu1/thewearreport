"""Acceptance tests for T-051: an attribute session with the answers taken live refuses to
start after dark in London unless --allow-dark is given. The task contract: do not edit.

Every frame here is synthetic (uniform colours with a gradient), the reviewers are
scripted, and every file is written by the tool into a temporary directory. Nothing
reaches the network.
"""

from __future__ import annotations

import ast
import datetime
import json
import socket
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wearreport import detect
from wearreport.tools import spotcheck

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "spotchecks" / "README.md"
UNIT_TESTS = ROOT / "engine" / "tests" / "unit" / "test_spotcheck.py"
H, W = 288, 352
DAY = datetime.date(2026, 10, 12)
NOON = datetime.datetime(2026, 10, 12, 11, 22, 33, tzinfo=datetime.UTC)
DUSK = datetime.datetime(2026, 9, 27, 18, 5, 59, tzinfo=datetime.UTC)  # civil twilight
NIGHT = datetime.datetime(2026, 10, 12, 21, 40, 0, tzinfo=datetime.UTC)
REFUSAL = (
    "spotcheck: it is dark in London now (sun below -6°); attribute sessions need "
    "daylight. Use --allow-dark to run anyway."
)
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
INFO = spotcheck.DetectorInfo(model="stub", sha256="0" * 64, conf=detect.DEFAULT_CONF)
WINDOW = ["--n", "5", "--min-persons", "1", "--view", "window"]

Frame = npt.NDArray[np.uint8]

# Boxes by height in source-frame pixels: 60 and 31 (near field), 30 (not).
B60 = (110.0, 20.0, 140.0, 80.0)
B31 = (60.0, 50.0, 80.0, 81.0)
B30 = (10.0, 50.0, 30.0, 80.0)
BOXES = [[B31, B30], [B60]]


# Helpers ------------------------------------------------------------------------------


class Stub:
    """boxes[i] are the person boxes in the frame whose colour is 10*i."""

    def __init__(self, boxes: Sequence[Sequence[tuple[float, float, float, float]]]) -> None:
        self.boxes = [list(b) for b in boxes]

    def detect(self, frame: Frame) -> list[detect.Detection]:
        return [detect.Detection("person", 0.9, b) for b in self.boxes[int(frame[0, 0, 0]) // 10]]


def _pipeline(
    boxes: Sequence[Sequence[tuple[float, float, float, float]]] = BOXES,
) -> spotcheck.Pipeline:
    frames = []
    for i in range(len(boxes)):
        frame = np.full((H, W, 3), 10 * i, dtype=np.uint8)
        frame[10:130, :, 1] = np.arange(W, dtype=np.uint8)[None, :]  # crops differ
        frame[0, 0, 0] = 10 * i
        frames.append(frame)
    return spotcheck.Pipeline(frames=lambda: frames, detector=Stub(boxes), info=INFO)


def _never() -> spotcheck.Pipeline:
    def frames() -> list[Frame]:
        raise AssertionError("the sweep must not start")

    return spotcheck.Pipeline(frames=frames, detector=Stub([]), info=INFO)


class Answers:
    """An attribute reviewer answering `default` for every crop; counts its calls."""

    def __init__(self, default: str | None = "ynn") -> None:
        self.default = default
        self.calls = 0

    def attributes(
        self, items: Sequence[spotcheck.ReviewItem], deadline: float
    ) -> dict[int, str | None]:
        self.calls += 1
        return {item.number: self.default for item in items}


class Pedestrians:
    """A detection reviewer calling every crop a pedestrian."""

    def judge(
        self, items: Sequence[spotcheck.ReviewItem], mode: str, deadline: float
    ) -> Mapping[int, spotcheck.Judgement]:
        return {item.number: spotcheck.Judgement(frozenset(), frozenset(), None) for item in items}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Outside CI, offline, in a fresh working directory, HOME and temporary directory;
    yields the temporary directory. The live and dry-run pipelines must not be opened."""
    for var in spotcheck.CI_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    work, home, tmp = tmp_path / "work", tmp_path / "home", tmp_path / "tmp"
    for d in (work, home, tmp):
        d.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)

    def no_network(self: socket.socket, address: Any) -> None:
        raise AssertionError(f"connection attempted: {address!r}")

    def no_pipeline(*args: Any, **kwargs: Any) -> spotcheck.Pipeline:
        raise AssertionError("the pipeline must not be opened")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(spotcheck, "live_pipeline", no_pipeline)
    monkeypatch.setattr(spotcheck, "dry_run_pipeline", no_pipeline)
    yield tmp


def _run(args: Sequence[str], out: Path, at: datetime.datetime, **kwargs: Any) -> int:
    argv = [*args, "--reviewer", "tester", "--out-dir", str(out)]
    return spotcheck.main(argv, today=DAY, clock=lambda: at, **kwargs)


def _record(out: Path) -> dict[str, Any]:
    path = out / "attributes" / f"{DAY.isoformat()}.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")) if root.exists() else []


# The clocks ---------------------------------------------------------------------------


def test_the_clocks_are_what_the_tests_say() -> None:
    assert spotcheck.light_at(NOON) == "day"
    assert spotcheck.light_at(DUSK) == "twilight"
    assert spotcheck.light_at(NIGHT) == "dark"


# AC1: the refusal ---------------------------------------------------------------------


def test_ac1_a_dark_live_session_is_refused(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    reviewer = Answers()
    args = ["--attributes", *WINDOW]
    assert _run(args, out, NIGHT, pipeline=_never(), reviewer=reviewer) == 1
    captured = capsys.readouterr()
    assert captured.err == REFUSAL + "\n"
    assert captured.out == ""
    assert not out.exists()
    assert reviewer.calls == 0
    assert _files(env) == []


def test_ac1_refused_before_the_pipeline_is_opened(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No pipeline given: the tool would open the live one (which the fixture forbids).
    out = tmp_path / "out"
    assert _run(["--attributes", *WINDOW], out, NIGHT, reviewer=Answers()) == 1
    assert capsys.readouterr().err == REFUSAL + "\n"
    assert not out.exists()
    assert _files(env) == []


def test_ac1_refused_before_the_dry_run_sweep(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    args = ["--attributes", "--dry-run", *WINDOW]
    assert _run(args, out, NIGHT, reviewer=Answers()) == 1
    assert capsys.readouterr().err == REFUSAL + "\n"
    assert not out.exists()


def test_ac1_the_injected_clock_decides(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Just before and after the -6 degree line on the same evening: 2026-10-12, sunset
    # about 17:10 UTC, civil dusk about 17:44 UTC in London.
    out = tmp_path / "out"
    minute = datetime.timedelta(minutes=1)
    dusk = NIGHT.replace(hour=17, minute=0)
    while spotcheck.light_at(dusk) != "dark":
        dusk += minute
    last_light = dusk - minute
    assert spotcheck.light_at(last_light) == "twilight"
    assert _run(["--attributes", *WINDOW], out, dusk, pipeline=_never(), reviewer=Answers()) == 1
    assert capsys.readouterr().err == REFUSAL + "\n"
    assert not out.exists()
    code = _run(
        ["--attributes", *WINDOW], out, last_light, pipeline=_pipeline(), reviewer=Answers()
    )
    assert code == 0
    assert _record(out)["light"] == "twilight"


# AC2: the override --------------------------------------------------------------------


def test_ac2_allow_dark_runs_a_normal_session(env: Path, tmp_path: Path) -> None:
    dark, day = tmp_path / "dark", tmp_path / "day"
    args = ["--attributes", *WINDOW]
    assert _run([*args, "--allow-dark"], dark, NIGHT, pipeline=_pipeline(), reviewer=Answers()) == 0
    assert _run(args, day, NOON, pipeline=_pipeline(), reviewer=Answers()) == 0
    at_night, at_noon = _record(dark), _record(day)
    assert at_night["light"] == "dark"
    assert at_night["started_at"] == "2026-10-12T21:40Z"
    assert at_noon["light"] == "day"
    for key in ("light", "started_at"):
        del at_night[key], at_noon[key]
    assert at_night == at_noon
    assert at_night["crops"] == [[31, "ynn", None], [60, "ynn", None]]


def test_ac2_allow_dark_without_attributes_is_refused(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    for at in (NOON, NIGHT):
        args = ["--n", "5", "--min-persons", "1", "--view", "window", "--allow-dark"]
        assert _run(args, out, at, pipeline=_never(), reviewer=Pedestrians()) == 1
        err = capsys.readouterr().err
        assert err.count("\n") == 1 and err.startswith("spotcheck: ")
        assert "--allow-dark" in err and "--attributes" in err
        assert not out.exists()


# AC3: unchanged elsewhere -------------------------------------------------------------


@pytest.mark.parametrize(("at", "light"), [(NOON, "day"), (DUSK, "twilight")])
def test_ac3_day_and_twilight_sessions_run_as_before(
    env: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    at: datetime.datetime,
    light: str,
) -> None:
    out = tmp_path / "out"
    assert _run(["--attributes", *WINDOW], out, at, pipeline=_pipeline(), reviewer=Answers()) == 0
    assert _record(out)["light"] == light
    assert "dark" not in capsys.readouterr().err


def test_ac3_a_judgements_file_runs_in_the_dark(env: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    answers = tmp_path / "answers.json"  # not read: the scripted reviewer answers
    args = ["--attributes", "--n", "5", "--min-persons", "1", "--judgements", str(answers)]
    assert _run(args, out, NIGHT, pipeline=_pipeline(), reviewer=Answers()) == 0
    record = _record(out)
    assert record["light"] == "dark"
    assert record["crops"] == [[31, "ynn", None], [60, "ynn", None]]


@pytest.mark.parametrize("record_boxes", [False, True])
def test_ac3_the_detection_check_runs_in_the_dark(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], record_boxes: bool
) -> None:
    dark, day = tmp_path / "dark", tmp_path / "day"
    args = ["--n", "5", "--min-persons", "1", "--view", "window"]
    if record_boxes:
        args.append("--record-boxes")
    assert _run(args, dark, NIGHT, pipeline=_pipeline(), reviewer=Pedestrians()) == 0
    assert "dark in London" not in capsys.readouterr().err
    assert _run(args, day, NOON, pipeline=_pipeline(), reviewer=Pedestrians()) == 0
    name = f"{DAY.isoformat()}.json"
    assert (dark / name).read_bytes() == (day / name).read_bytes()
    if record_boxes:
        boxes = json.loads((dark / "boxes" / name).read_text(encoding="utf-8"))
        assert boxes["light"] == "dark"


def test_ac3_the_refusal_names_the_same_threshold_as_the_light() -> None:
    assert spotcheck.LIGHT_TWILIGHT_DEG == -6


# AC4: wall-clock independence ---------------------------------------------------------


def test_ac4_the_unit_run_helper_passes_a_fixed_daytime_clock() -> None:
    tree = ast.parse(UNIT_TESTS.read_text(encoding="utf-8"))
    [helper] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run"
    ]
    [call] = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "setdefault"
    ]
    key, default = call.args
    assert isinstance(key, ast.Constant) and key.value == "clock"
    assert isinstance(default, ast.Lambda) and isinstance(default.body, ast.Name)
    name = default.body.id
    [value] = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]
    # A literal datetime.datetime(Y, M, D, h, m, s, tzinfo=datetime.UTC).
    assert isinstance(value, ast.Call) and ast.unparse(value.func) == "datetime.datetime"
    assert [(k.arg, ast.unparse(k.value)) for k in value.keywords] == [("tzinfo", "datetime.UTC")]
    fields = [int(ast.literal_eval(arg)) for arg in value.args] + [0, 0, 0]
    year, month, day, hour, minute, second = fields[:6]
    moment = datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.UTC)
    assert spotcheck.light_at(moment) == "day"


# AC5: docs ----------------------------------------------------------------------------


def test_ac5_readme_documents_the_refusal_and_the_flag() -> None:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Attribute session (`--attributes`)")
    end = text.index("\n## ", start + 1)
    section = text[start:end]
    assert "--allow-dark" in section
    assert "dark" in section and "-6°" in section
    assert "--judgements" in section
