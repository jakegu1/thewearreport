"""Acceptance tests for T-005 (sweep pipeline, aggregate schema and publisher). The task
contract: do not edit.

Tests that need a model file skip only when the file is missing and
WEARREPORT_REQUIRE_MODEL is unset; CI sets it, so there a missing model fails.
AC6 (growth) and AC8 (live dry run) are evidenced in the pull request.
"""

from __future__ import annotations

import copy
import json
import os
import re
import socket
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest

from wearreport import aggregate, cli, detect, publish, registry, weather
from wearreport.testing.fake_cameras import FakeCameraServer

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "data" / "schema" / "sweep.v1.json"
SAMPLE_PATH = ROOT / "data" / "schema" / "samples" / "sweep.v1.json"
PEOPLE = ROOT / "fixtures" / "detect" / "people_street.jpg"
REQUIRE_MODEL = "WEARREPORT_REQUIRE_MODEL"
RUNTIME_TEST = "engine/tests/unit/test_privacy_runtime.py"
ERROR_KINDS = ("timeout", "http", "decode", "network", "detect")
REQUIRED = {
    "sweep_id",
    "started_at",
    "finished_at",
    "source",
    "cameras_listed",
    "frames_ok",
    "frames_failed",
    "persons_total",
    "umbrellas_total",
    "per_camera",
    "weather",
    "engine_version",
    "model_sha256",
}
T0 = datetime(2026, 7, 15, 12, 30, 5, tzinfo=UTC)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(_load(SCHEMA_PATH))


def _model(name: str = "yolox_s.onnx") -> Path:
    path = detect.model_path(name)
    if not path.is_file():
        if os.environ.get(REQUIRE_MODEL):
            pytest.fail(f"{name} is missing and {REQUIRE_MODEL} is set")
        pytest.skip(f"{name} is missing; run make setup")
    return path


class _Blank:
    """A stand-in for YOLOX that finds nothing."""

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return [np.zeros(detect.OUTPUT_SHAPE, dtype=np.float32)]


def _blank_detector() -> detect.Detector:
    return detect.Detector.from_session(_Blank(), name="blank")


def _obs(camera_id: str, error: str | None = None, persons: int = 0, umbrellas: int = 0) -> Any:
    return aggregate.Observation(camera_id, error, persons, umbrellas)


def _record(
    started: datetime = T0,
    observations: list[Any] | None = None,
    conditions: weather.Conditions | None = None,
) -> dict[str, Any]:
    if observations is None:
        observations = [
            _obs("JamCams_00001.01251", persons=3, umbrellas=1),
            _obs("JamCams_00001.01252"),
            _obs("JamCams_00001.01253", "timeout"),
            _obs("JamCams_00001.01254", "detect"),
            _obs("JamCams_00001.01255", persons=1),
        ]
    return aggregate.build_record(
        observations,
        started_at=started,
        finished_at=started + timedelta(minutes=4, seconds=10),
        weather=conditions,
        engine_version="0.0.0",
        model_name="yolox_m",
        model_sha256=detect.MODEL_SHA256["yolox_m.onnx"],
    )


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Refuse any connection that is not to 127.0.0.1."""
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] == "127.0.0.1"):
            raise AssertionError(f"non-local connection attempted: {address!r}")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    yield


@pytest.fixture
def no_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sweep never runs git (or anything else)."""

    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError("the sweep spawned a process")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "system", refuse)


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeCameraServer]:
    """registry.list_cameras returns 6 cameras of a fake server: 2 show people, 2 fail."""
    for var in ("METOFFICE_API_KEY", "WEARREPORT_DEV_WEATHER", "WEARREPORT_ENV"):
        monkeypatch.delenv(var, raising=False)
    with FakeCameraServer() as server:
        cams = server.cameras(6)
        server.serve_body(cams[0].id, PEOPLE.read_bytes())
        server.serve_body(cams[1].id, PEOPLE.read_bytes())
        server.serve_404(cams[2].id)
        server.serve_corrupt(cams[3].id)

        def list_cameras(app_key: str | None, **kwargs: object) -> list[registry.Camera]:
            return cams

        monkeypatch.setattr(registry, "list_cameras", list_cameras)
        yield server


def _published(out: str) -> Path:
    match = re.search(r"^directory: (.+)$", out, re.MULTILINE)
    assert match, out
    return Path(match.group(1))


def _only_record(directory: Path) -> dict[str, Any]:
    files = sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())
    assert len(files) == 2 and "status.json" in files, files
    (name,) = [f for f in files if f != "status.json"]
    assert re.fullmatch(r"sweeps/\d{4}/\d{2}/\d{2}/\d{8}T\d{4}Z\.json", name), name
    record: dict[str, Any] = json.loads((directory / name).read_text(encoding="utf-8"))
    return record


# AC1: schema ---------------------------------------------------------------------------


def test_ac1_schema_is_draft_2020_12_and_requires_every_field() -> None:
    schema = _load(SCHEMA_PATH)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(schema)
    assert set(schema["required"]) >= REQUIRED
    assert schema["additionalProperties"] is False


def test_ac1_sample_validates(validator: jsonschema.Draft202012Validator) -> None:
    sample = _load(SAMPLE_PATH)
    validator.validate(sample)
    assert sample["source"] == "tfl-jamcam"
    assert sample["per_camera"], "the sample shows per-camera counts"


def _object_schemas(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if "properties" in node:
            yield node
        for value in node.values():
            yield from _object_schemas(value)
    elif isinstance(node, list):
        for value in node:
            yield from _object_schemas(value)


def test_ac1_additional_properties_false_at_every_level() -> None:
    schemas = list(_object_schemas(_load(SCHEMA_PATH)))
    assert len(schemas) >= 4  # record, frames_failed, a per-camera entry, weather
    for node in schemas:
        assert node.get("additionalProperties") is False, sorted(node["properties"])


@pytest.mark.parametrize(
    "path",
    [(), ("frames_failed",), ("per_camera", "JamCams_00001.01251"), ("weather",)],
    ids=["record", "frames_failed", "per_camera_entry", "weather"],
)
def test_ac1_an_extra_key_is_rejected_at_each_level(
    validator: jsonschema.Draft202012Validator, path: tuple[str, ...]
) -> None:
    conditions = weather.Conditions(14.5, 13.0, 0.4, T0.replace(minute=0, second=0), "metoffice")
    record = _record(conditions=conditions)
    validator.validate(record)
    node = record
    for key in path:
        node = node[key]
    node["box"] = [1, 2, 3, 4]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(record)


MUTATIONS: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
    ("sweep_id with seconds", lambda r: r.update(sweep_id="20260715T123005Z")),
    ("sweep_id without Z", lambda r: r.update(sweep_id="20260715T1230")),
    ("sweep_id with dashes", lambda r: r.update(sweep_id="2026-07-15T1230Z")),
    ("started_at not UTC", lambda r: r.update(started_at="2026-07-15T13:30:05+01:00")),
    ("finished_at naive", lambda r: r.update(finished_at="2026-07-15T12:34:15")),
    ("other source", lambda r: r.update(source="nyc-dot")),
    ("negative count", lambda r: r.update(persons_total=-1)),
    ("fractional count", lambda r: r.update(frames_ok=2.5)),
    ("unknown error category", lambda r: r["frames_failed"].update(teapot=1)),
    ("zero per-camera entry", lambda r: r["per_camera"].update(X={"persons": 0, "umbrellas": 0})),
    ("per-camera box", lambda r: r["per_camera"].update(X={"persons": 1, "box": [0, 0, 1, 1]})),
    ("model digest", lambda r: r.update(model_sha256="abc")),
    ("weather missing", lambda r: r.pop("weather")),
    ("weather string", lambda r: r.update(weather="sunny")),
]


@pytest.mark.parametrize("mutate", [m for _, m in MUTATIONS], ids=[n for n, _ in MUTATIONS])
def test_ac1_schema_rejects_invalid_records(
    validator: jsonschema.Draft202012Validator, mutate: Callable[[dict[str, Any]], None]
) -> None:
    record = _record()
    mutate(record)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(record)


def test_ac1_weather_holds_the_conditions_fields_or_null(
    validator: jsonschema.Draft202012Validator,
) -> None:
    observed = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    conditions = weather.Conditions(21.25, 20.0, 0.0, observed, "metoffice")
    record = _record(conditions=conditions)
    validator.validate(record)
    assert record["weather"] == {
        "temp_c": 21.25,
        "apparent_c": 20.0,
        "precip_mm": 0.0,
        "observed_at": "2026-07-15T12:00:00Z",
        "source": "metoffice",
    }
    none = _record()
    validator.validate(none)
    assert none["weather"] is None


# AC2: every record validates -----------------------------------------------------------


def test_ac2_built_record_validates_and_is_consistent(
    validator: jsonschema.Draft202012Validator,
) -> None:
    record = _record()
    validator.validate(record)
    assert record["sweep_id"] == "20260715T1230Z"
    assert record["started_at"] == "2026-07-15T12:30:05Z"
    assert record["finished_at"] == "2026-07-15T12:34:15Z"
    assert record["cameras_listed"] == 5
    assert record["frames_ok"] == 3
    assert record["frames_failed"] == {k: int(k in ("timeout", "detect")) for k in ERROR_KINDS}
    assert record["persons_total"] == 4 and record["umbrellas_total"] == 1
    assert record["per_camera"] == {
        "JamCams_00001.01251": {"persons": 3, "umbrellas": 1},
        "JamCams_00001.01255": {"persons": 1, "umbrellas": 0},
    }
    aggregate.check_record(record)


def test_ac2_sweep_with_zero_cameras_or_all_failures_validates(
    validator: jsonschema.Draft202012Validator,
) -> None:
    empty = _record(observations=[])
    validator.validate(empty)
    assert empty["cameras_listed"] == 0 and empty["per_camera"] == {}
    failed = _record(observations=[_obs(f"C{i}", "network") for i in range(4)])
    validator.validate(failed)
    assert failed["frames_ok"] == 0 and failed["frames_failed"]["network"] == 4


def test_ac2_pipeline_record_validates(
    validator: jsonschema.Draft202012Validator,
    fake_registry: FakeCameraServer,
    offline: None,
) -> None:
    record = aggregate.run_sweep(_blank_detector(), "0" * 64, model_name="blank")
    validator.validate(record)
    assert record["cameras_listed"] == 6
    assert record["frames_ok"] == 4
    assert record["frames_failed"]["http"] == 1 and record["frames_failed"]["decode"] == 1
    assert record["per_camera"] == {} and record["persons_total"] == 0
    assert record["weather"] is None  # no Met Office key


def test_ac2_pipeline_with_the_model_counts_people(
    validator: jsonschema.Draft202012Validator,
    fake_registry: FakeCameraServer,
    offline: None,
) -> None:
    path = _model()
    record = aggregate.run_sweep(
        detect.Detector(path), detect.sha256_of(path), model_name="yolox_s"
    )
    validator.validate(record)
    assert set(record["per_camera"]) == {"Fake_00001", "Fake_00002"}
    assert record["persons_total"] >= 2
    assert record["persons_total"] == sum(c["persons"] for c in record["per_camera"].values())


# AC3 and AC4: idempotent, append-only publishing ---------------------------------------


def test_ac3_same_sweep_twice_stores_one_record(tmp_path: Path) -> None:
    record = _record()
    first = publish.publish(tmp_path, record, now=T0 + timedelta(minutes=5))
    before = _files(tmp_path)
    second = publish.publish(tmp_path, copy.deepcopy(record), now=T0 + timedelta(minutes=5))
    assert first.created and not second.created
    assert first.record_path == second.record_path
    assert _files(tmp_path) == before
    assert len([f for f in before if f.startswith("sweeps/")]) == 1


def test_ac3_same_sweep_id_with_other_content_raises_and_writes_nothing(tmp_path: Path) -> None:
    record = _record()
    publish.publish(tmp_path, record, now=T0 + timedelta(minutes=5))
    before = _files(tmp_path)
    stat = os.stat(tmp_path / "status.json")
    other = _record(started=T0 + timedelta(seconds=30))  # same minute, same sweep_id
    assert other["sweep_id"] == record["sweep_id"] and other != record
    with pytest.raises(publish.PublishError):
        publish.publish(tmp_path, other, now=T0 + timedelta(minutes=6))
    assert _files(tmp_path) == before
    assert os.stat(tmp_path / "status.json").st_mtime_ns == stat.st_mtime_ns
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == sorted(
        [Path(f).name for f in before]
    )


def test_ac3_invalid_record_is_refused_before_writing(tmp_path: Path) -> None:
    record = _record()
    record["per_camera"]["JamCams_00001.01251"]["box"] = [1, 2, 3, 4]
    with pytest.raises((publish.PublishError, aggregate.RecordError)):
        publish.publish(tmp_path, record, now=T0)
    assert list(tmp_path.iterdir()) == []


def test_ac4_layout_and_records_are_never_rewritten(tmp_path: Path) -> None:
    times = [T0 - timedelta(hours=13), T0 - timedelta(minutes=20), T0]
    paths = []
    for started in times:
        result = publish.publish(
            tmp_path, _record(started=started), now=started + timedelta(minutes=5)
        )
        paths.append(result.record_path)
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == [
        "sweeps/2026/07/14/20260714T2330Z.json",
        "sweeps/2026/07/15/20260715T1210Z.json",
        "sweeps/2026/07/15/20260715T1230Z.json",
    ]
    first = paths[0]
    inode, mtime = first.stat().st_ino, first.stat().st_mtime_ns
    content = first.read_bytes()
    publish.publish(tmp_path, _record(started=T0 + timedelta(hours=1)), now=T0 + timedelta(hours=2))
    assert first.stat().st_ino == inode and first.stat().st_mtime_ns == mtime
    assert first.read_bytes() == content
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert len(files) == 5 and "status.json" in files


def test_ac4_record_file_is_one_json_line(tmp_path: Path) -> None:
    result = publish.publish(tmp_path, _record(), now=T0 + timedelta(minutes=5))
    text = result.record_path.read_text(encoding="utf-8")
    assert text.endswith("\n") and text.count("\n") == 1
    assert json.loads(text) == _record()


# AC5: status.json ----------------------------------------------------------------------


def _write(data_dir: Path, started: datetime, listed: int, ok: int, persons: int) -> None:
    observations = [_obs(f"C{i:04d}", None if i < ok else "timeout") for i in range(listed)]
    if persons:
        observations[0] = _obs("C0000", None, persons=persons)
    record = _record(started=started, observations=observations)
    path = publish.record_path(data_dir, record["sweep_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(publish.serialize(record))


def test_ac5_status_in_summer_crosses_midnight_and_the_london_day(tmp_path: Path) -> None:
    now = datetime(2026, 7, 15, 1, 0, tzinfo=UTC)  # 02:00 BST
    rows = [
        (datetime(2026, 7, 14, 0, 30, tzinfo=UTC), 10, 10, 99),  # older than 24 h
        (datetime(2026, 7, 14, 1, 0, tzinfo=UTC), 10, 10, 98),  # exactly 24 h: outside
        (datetime(2026, 7, 14, 5, 59, tzinfo=UTC), 10, 10, 50),  # 06:59 BST: not daytime
        (datetime(2026, 7, 14, 6, 0, tzinfo=UTC), 10, 9, 4),  # 07:00 BST, 90%: success
        (datetime(2026, 7, 14, 12, 0, tzinfo=UTC), 10, 10, 6),  # daytime
        (datetime(2026, 7, 14, 19, 59, tzinfo=UTC), 10, 10, 8),  # 20:59 BST: daytime
        (datetime(2026, 7, 14, 20, 0, tzinfo=UTC), 10, 10, 70),  # 21:00 BST: not daytime
        (datetime(2026, 7, 14, 23, 30, tzinfo=UTC), 10, 8, 0),  # 80%: failure
        (datetime(2026, 7, 15, 0, 30, tzinfo=UTC), 1000, 899, 0),  # 89.9%: failure
    ]
    for started, listed, ok, persons in rows:
        _write(tmp_path, started, listed, ok, persons)
    status = publish.compute_status(tmp_path, now=now)
    assert status["last_sweep_at"] == "2026-07-15T00:30:00Z"
    assert status["sweeps_24h"] == 7
    assert status["success_rate_24h"] == pytest.approx(5 / 7)
    assert status["median_persons_daytime_24h"] == 6
    assert status["consecutive_failures"] == 2


def test_ac5_status_in_winter_uses_gmt(tmp_path: Path) -> None:
    now = datetime(2026, 1, 15, 22, 0, tzinfo=UTC)
    rows = [
        (datetime(2026, 1, 15, 6, 30, tzinfo=UTC), 10, 10, 40),  # 06:30 GMT: not daytime
        (datetime(2026, 1, 15, 7, 0, tzinfo=UTC), 10, 10, 3),  # 07:00 GMT: daytime
        (datetime(2026, 1, 15, 20, 30, tzinfo=UTC), 10, 10, 5),  # 20:30 GMT: daytime
        (datetime(2026, 1, 15, 21, 0, tzinfo=UTC), 10, 10, 60),  # 21:00 GMT: not daytime
    ]
    for started, listed, ok, persons in rows:
        _write(tmp_path, started, listed, ok, persons)
    status = publish.compute_status(tmp_path, now=now)
    assert status["sweeps_24h"] == 4
    assert status["success_rate_24h"] == 1
    assert status["median_persons_daytime_24h"] == 4
    assert status["consecutive_failures"] == 0


def test_ac5_status_with_no_records(tmp_path: Path) -> None:
    status = publish.compute_status(tmp_path, now=T0)
    assert status["last_sweep_at"] is None
    assert status["sweeps_24h"] == 0
    assert status["success_rate_24h"] is None
    assert status["median_persons_daytime_24h"] is None
    assert status["consecutive_failures"] == 0


def test_ac5_consecutive_failures_reach_past_24_hours(tmp_path: Path) -> None:
    _write(tmp_path, T0 - timedelta(days=3), 10, 10, 1)  # the last success
    for hours in (60, 40, 20, 2):
        _write(tmp_path, T0 - timedelta(hours=hours), 10, 0, 0)
    _write(tmp_path, T0 - timedelta(hours=1), 0, 0, 0)  # no cameras listed: a failure
    status = publish.compute_status(tmp_path, now=T0)
    assert status["consecutive_failures"] == 5
    assert status["sweeps_24h"] == 3 and status["success_rate_24h"] == 0


def test_ac5_status_json_is_computed_from_the_record_files(tmp_path: Path) -> None:
    publish.publish(tmp_path, _record(started=T0 - timedelta(hours=1)), now=T0 - timedelta(hours=1))
    _write(tmp_path, T0 - timedelta(minutes=30), 10, 10, 7)  # a record written by hand
    publish.publish(tmp_path, _record(started=T0), now=T0 + timedelta(minutes=5))
    written = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    computed = publish.compute_status(tmp_path, now=T0 + timedelta(minutes=5))
    assert written == json.loads(json.dumps(computed))
    assert written["sweeps_24h"] == 3
    assert written["last_sweep_at"] == "2026-07-15T12:30:05Z"


# AC7: command line ---------------------------------------------------------------------


def test_ac7_console_script_is_declared() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["scripts"]["wearreport"] == "wearreport.cli:main"


def test_ac7_dry_run_writes_record_and_status_to_a_new_temporary_directory(
    fake_registry: FakeCameraServer,
    offline: None,
    no_subprocess: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    validator: jsonschema.Draft202012Validator,
) -> None:
    _model()
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["sweep", "--dry-run", "--model", "yolox_s.onnx"]) == 0
    directory = _published(capsys.readouterr().out)
    assert directory.parent == tmp_path and directory.is_dir()
    record = _only_record(directory)
    validator.validate(record)
    assert record["model_sha256"] == detect.MODEL_SHA256["yolox_s.onnx"]
    assert record["cameras_listed"] == 6 and record["persons_total"] >= 2
    assert [p for p in tmp_path.iterdir()] == [directory]


def test_ac7_data_dir_writes_into_the_checkout_without_committing(
    fake_registry: FakeCameraServer,
    offline: None,
    no_subprocess: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model()
    monkeypatch.setattr(detect, "open_session", lambda path: _Blank())
    checkout = tmp_path / "data"
    (checkout / ".git").mkdir(parents=True)
    (checkout / ".git" / "HEAD").write_text("ref: refs/heads/data\n")
    assert cli.main(["sweep", "--data-dir", str(checkout), "--model", "yolox_s.onnx"]) == 0
    assert (checkout / ".git" / "HEAD").read_text() == "ref: refs/heads/data\n"
    assert sorted(p.name for p in (checkout / ".git").iterdir()) == ["HEAD"]
    assert (checkout / "status.json").is_file()
    assert len(list((checkout / "sweeps").rglob("*.json"))) == 1


def test_ac7_default_model_is_yolox_m_and_its_digest_is_recorded(
    fake_registry: FakeCameraServer,
    offline: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model("yolox_m.onnx")
    loaded: list[Path] = []

    def open_session(path: Path) -> _Blank:
        loaded.append(Path(path))
        return _Blank()

    monkeypatch.setattr(detect, "open_session", open_session)
    assert cli.main(["sweep", "--data-dir", str(tmp_path)]) == 0
    assert [p.name for p in loaded] == ["yolox_m.onnx"]
    (record,) = [json.loads(p.read_text()) for p in (tmp_path / "sweeps").rglob("*.json")]
    assert record["model_sha256"] == detect.MODEL_SHA256["yolox_m.onnx"]


def test_ac7_unknown_model_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["sweep", "--data-dir", str(tmp_path), "--model", "../evil.onnx"])
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_ac7_weather_comes_from_current_conditions(
    fake_registry: FakeCameraServer,
    offline: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model()
    monkeypatch.setattr(detect, "open_session", lambda path: _Blank())
    calls: list[datetime] = []
    observed = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)

    def current_conditions(at: datetime, **kwargs: object) -> weather.Conditions:
        calls.append(at)
        return weather.Conditions(18.0, 17.5, 1.2, observed, "metoffice")

    def direct(*args: object, **kwargs: object) -> None:
        raise AssertionError("a provider was called directly, bypassing the request cap")

    monkeypatch.setattr(weather, "current_conditions", current_conditions)
    monkeypatch.setattr(weather.MetOfficeProvider, "fetch", direct)
    monkeypatch.setattr(weather.OpenMeteoProvider, "fetch", direct)
    assert cli.main(["sweep", "--data-dir", str(tmp_path), "--model", "yolox_s.onnx"]) == 0
    assert len(calls) == 1 and calls[0].tzinfo is not None
    (record,) = [json.loads(p.read_text()) for p in (tmp_path / "sweeps").rglob("*.json")]
    assert record["weather"]["temp_c"] == 18.0
    assert record["weather"]["source"] == "metoffice"
    assert record["started_at"] <= f"{calls[0].astimezone(UTC):%Y-%m-%dT%H:%M:%S}Z"


def test_ac7_weather_config_error_stops_the_sweep_and_writes_nothing(
    fake_registry: FakeCameraServer,
    offline: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _model()
    monkeypatch.setenv("WEARREPORT_ENV", "production")
    monkeypatch.setenv("WEARREPORT_DEV_WEATHER", "openmeteo")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr("tempfile.tempdir", None)
    data = tmp_path / "data"
    data.mkdir()
    assert cli.main(["sweep", "--data-dir", str(data), "--model", "yolox_s.onnx"]) != 0
    assert "weather" in capsys.readouterr().err.lower()
    assert list(data.iterdir()) == []
    assert cli.main(["sweep", "--dry-run", "--model", "yolox_s.onnx"]) != 0
    assert list((tmp_path / "tmp").iterdir()) == []
    assert fake_registry.requests("Fake_00001") == 0  # stopped before any camera


# AC9: runtime privacy -------------------------------------------------------------------


def _run_runtime_test(canary: bool) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "WEARREPORT_PRIVACY_CANARY"}
    if canary:
        env["WEARREPORT_PRIVACY_CANARY"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-k",
            "pipeline",
            RUNTIME_TEST,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_ac9_runtime_privacy_test_runs_the_sweep_command() -> None:
    source = (ROOT / RUNTIME_TEST).read_text()
    assert "def test_pipeline" in source
    assert '"sweep", "--dry-run"' in source


def test_ac9_pipeline_privacy_tests_pass_without_skipping() -> None:
    proc = _run_runtime_test(canary=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    match = re.search(r"\b(\d+) passed\b", proc.stdout)
    assert match and int(match.group(1)) >= 2, proc.stdout
    assert "skipped" not in proc.stdout


def test_ac9_pipeline_privacy_test_fails_when_the_canary_hook_is_enabled() -> None:
    proc = _run_runtime_test(canary=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "failed" in proc.stdout
