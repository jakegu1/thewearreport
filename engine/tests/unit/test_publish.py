"""The publisher, status.json and the `wearreport sweep` command (wearreport.publish, .cli)."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from wearreport import aggregate, cli, detect, publish, registry

T0 = datetime(2026, 7, 15, 12, 30, 5, tzinfo=UTC)


def _record(started: datetime = T0, ok: int = 10, listed: int = 10, persons: int = 3) -> Any:
    observations = [
        aggregate.Observation(f"C{i:04d}", None if i < ok else "timeout") for i in range(listed)
    ]
    if persons and ok:
        observations[0] = aggregate.Observation("C0000", None, persons, 0)
    return aggregate.build_record(
        observations,
        started_at=started,
        finished_at=started + timedelta(minutes=4),
        weather=None,
        engine_version="0.0.0",
        model_name="yolox_m",
        model_sha256="b" * 64,
    )


def _put(data_dir: Path, record: dict[str, Any]) -> Path:
    """Write a record file directly, as a checkout of the data branch would hold it."""
    path = publish.record_path(data_dir, record["sweep_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(publish.serialize(record))
    return path


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


# Serialisation and layout --------------------------------------------------------------


def test_serialize_is_canonical_ascii_json_on_one_line() -> None:
    record = _record()
    data = publish.serialize(record)
    assert data.endswith(b"\n") and data.count(b"\n") == 1
    assert b": " not in data and b", " not in data
    assert data == publish.serialize(dict(reversed(list(record.items()))))
    assert json.loads(data) == record


def test_serialize_refuses_nan() -> None:
    with pytest.raises(ValueError):
        publish.serialize({"x": float("nan")})


@pytest.mark.parametrize("sweep_id", ["", "20260715T1230", "../../etc/passwd", "20260715T1230Z/.."])
def test_record_path_refuses_malformed_ids(tmp_path: Path, sweep_id: str) -> None:
    with pytest.raises(aggregate.RecordError):
        publish.record_path(tmp_path, sweep_id)


def test_record_path_uses_the_utc_date(tmp_path: Path) -> None:
    path = publish.record_path(tmp_path, "20261231T2359Z")
    assert path == tmp_path / "sweeps" / "2026" / "12" / "31" / "20261231T2359Z.json"


# Publishing ----------------------------------------------------------------------------


def test_publish_into_a_missing_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(publish.PublishError):
        publish.publish(tmp_path / "absent", _record(), now=T0)
    assert not (tmp_path / "absent").exists()


def test_publish_into_a_file_fails(tmp_path: Path) -> None:
    target = tmp_path / "file"
    target.write_text("x")
    with pytest.raises(publish.PublishError):
        publish.publish(target, _record(), now=T0)


def test_publish_leaves_no_temporary_files(tmp_path: Path) -> None:
    publish.publish(tmp_path, _record(), now=T0)
    publish.publish(tmp_path, _record(), now=T0)
    with pytest.raises(publish.ConflictError):
        publish.publish(tmp_path, _record(persons=4), now=T0)
    assert _tree(tmp_path) == [
        "status.json",
        "sweeps",
        "sweeps/2026",
        "sweeps/2026/07",
        "sweeps/2026/07/15",
        "sweeps/2026/07/15/20260715T1230Z.json",
    ]


def test_publish_refuses_a_symlinked_directory(tmp_path: Path) -> None:
    data, outside = tmp_path / "data", tmp_path / "outside"
    data.mkdir()
    outside.mkdir()
    (data / "sweeps").symlink_to(outside)
    with pytest.raises(publish.PublishError):
        publish.publish(data, _record(), now=T0)
    assert list(outside.iterdir()) == []
    assert not (data / "status.json").exists()


def test_publish_refuses_an_existing_record_reached_through_a_symlink(tmp_path: Path) -> None:
    data, outside = tmp_path / "data", tmp_path / "outside"
    data.mkdir()
    _put(outside, _record())
    (data / "sweeps").symlink_to(outside / "sweeps")
    with pytest.raises(publish.PublishError):
        publish.publish(data, _record(), now=T0)


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_publish_refuses_a_record_path_that_is_not_a_file(tmp_path: Path, kind: str) -> None:
    path = publish.record_path(tmp_path, _record()["sweep_id"])
    path.parent.mkdir(parents=True)
    if kind == "directory":
        path.mkdir()
    else:
        (tmp_path / "elsewhere.json").write_bytes(publish.serialize(_record()))
        path.symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(publish.PublishError):
        publish.publish(tmp_path, _record(), now=T0)
    assert not (tmp_path / "status.json").exists()


def test_publish_refuses_an_invalid_record_and_writes_nothing(tmp_path: Path) -> None:
    record = _record()
    record["persons_total"] += 1  # no longer the sum over per_camera
    with pytest.raises(aggregate.RecordError):
        publish.publish(tmp_path, record, now=T0)
    assert list(tmp_path.iterdir()) == []


def test_publish_replaces_status_but_keeps_records(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text("not json\n")
    first = publish.publish(tmp_path, _record(), now=T0)
    status_before = (tmp_path / "status.json").read_bytes()
    second = publish.publish(
        tmp_path, _record(T0 + timedelta(hours=1)), now=T0 + timedelta(hours=1)
    )
    assert first.record_path.read_bytes() == publish.serialize(_record())
    assert json.loads(status_before)["sweeps_24h"] == 1
    assert json.loads(second.status_path.read_bytes())["sweeps_24h"] == 2


def test_publish_fails_loudly_when_status_cannot_be_written(tmp_path: Path) -> None:
    (tmp_path / "status.json").mkdir()
    with pytest.raises(publish.PublishError):
        publish.publish(tmp_path, _record(), now=T0)
    # The record is published and stays: records are never deleted.
    assert publish.record_path(tmp_path, _record()["sweep_id"]).is_file()


def _race(tmp_path: Path, records: list[Any]) -> tuple[list[bool], list[BaseException]]:
    created: list[bool] = []
    errors: list[BaseException] = []
    start = threading.Barrier(len(records))

    def worker(record: Any) -> None:
        start.wait()
        try:
            created.append(publish.publish(tmp_path, record, now=T0).created)
        except publish.PublishError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(r,)) for r in records]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return created, errors


def test_concurrent_identical_publishes_create_one_record(tmp_path: Path) -> None:
    created, errors = _race(tmp_path, [_record() for _ in range(8)])
    assert errors == []
    assert sorted(created) == [False] * 7 + [True]
    assert len(list(tmp_path.rglob("*.json"))) == 2  # the record and status.json
    assert not [p for p in tmp_path.rglob(".*")]


def test_concurrent_conflicting_publishes_keep_the_first(tmp_path: Path) -> None:
    records = [_record(persons=p) for p in range(1, 9)]
    created, errors = _race(tmp_path, records)
    assert created == [True] and len(errors) == 7
    assert all(isinstance(e, publish.ConflictError) for e in errors)
    stored = json.loads(publish.record_path(tmp_path, records[0]["sweep_id"]).read_text())
    assert stored in records
    assert not [p for p in tmp_path.rglob(".*")]


# status.json ---------------------------------------------------------------------------


def test_is_success_threshold() -> None:
    assert publish.is_success(_record(ok=9, listed=10))
    assert not publish.is_success(_record(ok=8, listed=10))
    assert publish.is_success(_record(ok=900, listed=1000))
    assert not publish.is_success(_record(ok=899, listed=1000))
    assert not publish.is_success(_record(ok=0, listed=0))


@pytest.mark.parametrize(
    ("moment", "daytime"),
    [
        (datetime(2026, 3, 28, 6, 30, tzinfo=UTC), False),  # 06:30 GMT, the day before BST
        (datetime(2026, 3, 29, 6, 30, tzinfo=UTC), True),  # 07:30 BST, the first BST day
        (datetime(2026, 3, 29, 20, 0, tzinfo=UTC), False),  # 21:00 BST
        (datetime(2026, 10, 25, 6, 30, tzinfo=UTC), False),  # 06:30 GMT, the first GMT day
        (datetime(2026, 10, 25, 7, 0, tzinfo=UTC), True),  # 07:00 GMT
        (datetime(2026, 10, 24, 19, 59, tzinfo=UTC), True),  # 20:59 BST
        (datetime(2026, 10, 25, 20, 59, tzinfo=UTC), True),  # 20:59 GMT
        (datetime(2026, 10, 25, 21, 0, tzinfo=UTC), False),
        (datetime.min.replace(tzinfo=UTC), False),  # no overflow at the edge of the range
    ],
)
def test_london_daytime_follows_the_clock_changes(moment: datetime, daytime: bool) -> None:
    assert publish.is_london_daytime(moment) is daytime


def test_status_across_the_autumn_clock_change(tmp_path: Path) -> None:
    now = datetime(2026, 10, 25, 22, 0, tzinfo=UTC)
    for hour, persons in [(6, 100), (7, 2), (12, 4), (20, 6), (21, 100)]:
        _put(tmp_path, _record(datetime(2026, 10, 25, hour, 0, tzinfo=UTC), persons=persons))
    status = publish.compute_status(tmp_path, now=now)
    assert status["sweeps_24h"] == 5 and status["daytime_sweeps_24h"] == 3
    assert status["median_persons_daytime_24h"] == 4


def test_status_median_of_an_even_count(tmp_path: Path) -> None:
    for minute, persons in [(0, 3), (10, 4)]:
        _put(tmp_path, _record(T0.replace(minute=minute), persons=persons))
    assert publish.compute_status(tmp_path, now=T0)["median_persons_daytime_24h"] == 3.5


def test_status_ignores_records_from_the_future(tmp_path: Path) -> None:
    _put(tmp_path, _record(T0 - timedelta(hours=1)))
    _put(tmp_path, _record(T0 + timedelta(days=1), ok=0))
    status = publish.compute_status(tmp_path, now=T0)
    assert status["last_sweep_at"] == "2026-07-15T11:30:05Z"
    assert status["sweeps_24h"] == 1 and status["consecutive_failures"] == 0


def test_status_carries_the_attribution_statements(tmp_path: Path) -> None:
    status = publish.compute_status(tmp_path, now=T0)
    assert status["attribution"] == ["Powered by TfL Open Data", "Powered by Met Office data"]
    assert status["generated_at"] == "2026-07-15T12:30:05Z"


HOSTILE: list[tuple[str, bytes]] = [
    ("not json", b"{not json"),
    ("invalid utf-8", b'{"a": "\xff"}'),
    ("deep nesting", b"[" * 100_000 + b"]" * 100_000),
    ("huge integer", b'{"n": ' + b"9" * 5000 + b"}"),
    ("too large", b" " * (publish.MAX_RECORD_BYTES + 1)),
    ("empty", b""),
    ("a list", b"[]\n"),
]


@pytest.mark.parametrize("body", [b for _, b in HOSTILE], ids=[n for n, _ in HOSTILE])
def test_status_counts_and_skips_hostile_record_files(tmp_path: Path, body: bytes) -> None:
    good = _put(tmp_path, _record(T0 - timedelta(hours=2)))
    bad = good.parent / "20260715T1100Z.json"
    bad.write_bytes(body)
    status = publish.compute_status(tmp_path, now=T0)
    assert status["records_invalid"] == 1
    assert status["sweeps_24h"] == 1
    assert status["last_sweep_id"] == "20260715T1030Z"


def test_status_skips_records_filed_in_the_wrong_place(tmp_path: Path) -> None:
    record = _record(T0 - timedelta(hours=1))
    wrong_name = publish.record_path(tmp_path, "20260715T1100Z")
    wrong_name.parent.mkdir(parents=True)
    wrong_name.write_bytes(publish.serialize(record))  # the file of another sweep_id
    wrong_day = tmp_path / "sweeps" / "2026" / "07" / "14" / f"{record['sweep_id']}.json"
    wrong_day.parent.mkdir(parents=True)
    wrong_day.write_bytes(publish.serialize(record))
    status = publish.compute_status(tmp_path, now=T0)
    assert status["records_invalid"] == 2 and status["sweeps_24h"] == 0


def test_status_ignores_symlinks_and_unrelated_files(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _put(outside, _record(T0 - timedelta(hours=1)))
    data = tmp_path / "data"
    (data / "sweeps").mkdir(parents=True)
    (data / "sweeps" / "2026").symlink_to(outside / "sweeps" / "2026")
    (data / "sweeps" / "README.md").write_text("records live below\n")
    day = data / "sweeps" / "2025" / "07" / "15"
    day.mkdir(parents=True)
    (day / ".20250715T1230Z.json.0123456789abcdef.tmp").write_text("{}")
    (day / "20250715T1230Z.json").symlink_to(publish.record_path(outside, _record()["sweep_id"]))
    status = publish.compute_status(data, now=T0)
    assert status["sweeps_24h"] == 0 and status["last_sweep_at"] is None
    assert status["records_invalid"] == 1  # the symlink named like a record


def test_status_stops_reading_once_past_the_window_and_a_success(tmp_path: Path) -> None:
    _put(tmp_path, _record(T0 - timedelta(hours=1)))
    _put(tmp_path, _record(T0 - timedelta(days=2)))  # past the window, after a success
    old = tmp_path / "sweeps" / "2025" / "01" / "01"
    old.mkdir(parents=True)
    (old / "20250101T1200Z.json").write_text("{broken")  # never read
    assert publish.compute_status(tmp_path, now=T0)["records_invalid"] == 0


# The command line ----------------------------------------------------------------------


class _Blank:
    def run(self, tensor: Any) -> list[Any]:
        import numpy as np

        return [np.zeros(detect.OUTPUT_SHAPE, dtype=np.float32)]


@pytest.fixture
def no_cameras(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in ("METOFFICE_API_KEY", "WEARREPORT_DEV_WEATHER", "WEARREPORT_ENV"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(registry, "list_cameras", lambda app_key, **kwargs: [])
    monkeypatch.setattr(detect, "open_session", lambda path: _Blank())
    yield


def _model_or_skip() -> None:
    if not detect.model_path("yolox_s.onnx").is_file():
        if os.environ.get("WEARREPORT_REQUIRE_MODEL"):
            pytest.fail("yolox_s.onnx is missing and WEARREPORT_REQUIRE_MODEL is set")
        pytest.skip("yolox_s.onnx is missing; run make setup")


def test_cli_refuses_a_missing_data_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["sweep", "--data-dir", str(tmp_path / "absent")]) == 1
    assert "does not exist" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_cli_needs_exactly_one_target(tmp_path: Path) -> None:
    for argv in (["sweep"], ["sweep", "--dry-run", "--data-dir", str(tmp_path)], []):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2


def test_cli_reports_a_registry_failure_and_writes_nothing(
    no_cameras: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _model_or_skip()

    def down(app_key: str | None, **kwargs: object) -> list[registry.Camera]:
        raise registry.RegistryError("JamCam registry unavailable after 3 attempts (URLError)")

    monkeypatch.setattr(registry, "list_cameras", down)
    assert cli.main(["sweep", "--data-dir", str(tmp_path), "--model", "yolox_s.onnx"]) == 1
    assert "registry unavailable" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_cli_publishes_a_sweep_of_zero_cameras(
    no_cameras: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _model_or_skip()
    assert cli.main(["sweep", "--data-dir", str(tmp_path), "--model", "yolox_s.onnx"]) == 0
    out = capsys.readouterr().out
    assert "cameras listed: 0" in out and "weather: none" in out
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["consecutive_failures"] == 1 and status["success_rate_24h"] == 0


def test_cli_reports_a_conflicting_sweep_id(
    no_cameras: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _model_or_skip()
    argv = ["sweep", "--data-dir", str(tmp_path), "--model", "yolox_s.onnx"]
    monkeypatch.setattr(aggregate, "_utc_now", lambda: T0)
    assert cli.main(argv) == 0
    assert cli.main(argv) == 0  # the same sweep again: nothing changes
    assert "(already published)" in capsys.readouterr().out
    (record_file,) = (tmp_path / "sweeps").rglob("*.json")
    stored = record_file.read_bytes()
    monkeypatch.setattr(aggregate, "engine_version", lambda: "0.0.1")  # other content
    assert cli.main(argv) == 1
    assert "already published" in capsys.readouterr().err
    assert record_file.read_bytes() == stored


def test_cli_refuses_a_model_file_that_is_not_its_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _model_or_skip()
    fake_m = tmp_path / "yolox_m.onnx"
    fake_m.symlink_to(detect.model_path("yolox_s.onnx"))  # a pinned model, but not YOLOX-m
    monkeypatch.setattr(detect, "MODEL_DIR", tmp_path)
    assert cli.main(["sweep", "--data-dir", str(tmp_path)]) == 1
    assert "does not match its pinned SHA-256" in capsys.readouterr().err


def test_cli_reports_a_missing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(detect, "MODEL_DIR", tmp_path / "none")
    assert cli.main(["sweep", "--data-dir", str(tmp_path)]) == 1
    assert "make setup" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_cli_refuses_a_model_that_changes_while_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _model_or_skip()
    digests = iter([detect.MODEL_SHA256["yolox_s.onnx"], "0" * 64])
    monkeypatch.setattr(detect, "sha256_of", lambda path: next(digests))
    monkeypatch.setattr(detect, "open_session", lambda path: _Blank())
    assert cli.main(["sweep", "--data-dir", str(tmp_path), "--model", "yolox_s.onnx"]) == 1
    assert "changed while it was being loaded" in capsys.readouterr().err


def test_json_log_lines_carry_extra_fields() -> None:
    import logging

    record = logging.makeLogRecord({"msg": "sweep published", "levelname": "INFO"})
    record.sweep_id = "20260715T1230Z"
    line = json.loads(cli._JsonFormatter().format(record))
    assert line["message"] == "sweep published" and line["sweep_id"] == "20260715T1230Z"
    assert "args" not in line and "exc_info" not in line


def test_cli_reports_a_record_that_cannot_be_built(
    no_cameras: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _model_or_skip()
    monkeypatch.setattr(aggregate, "engine_version", lambda: "not a version")
    assert cli.main(["sweep", "--data-dir", str(tmp_path), "--model", "yolox_s.onnx"]) == 1
    assert "engine_version is malformed" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
