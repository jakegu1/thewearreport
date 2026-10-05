"""Acceptance tests for T-067: the one-click labelling launcher for Windows (`label.cmd`,
`scripts/label.ps1`) and its helper (`wearreport.tools.label_window`). The task contract:
do not edit.

On Linux these tests check the helper that picks the city and the next daylight window,
with fixed clocks, and the launcher's files as text. On Windows (the windows-latest CI
job) the launcher itself runs end to end with its test switches: spotcheck's dry-run
pipeline (the licensed fixture photos, served on this machine), the `--judgements` file
reviewer answered by a test script, a fake judge on 127.0.0.1, a fake key, fake `git` and
`uv` on PATH and a file in place of the clipboard. Nothing reaches the network, and no
image is opened or looked at: the test reviewer reads only the review's numbering file.
"""

from __future__ import annotations

import ast
import datetime
import json
import os
import re
import secrets
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from wearreport.tools import judge_hosted, label_window, pilot_heights, spotcheck

ROOT = Path(__file__).resolve().parents[3]
LABEL_CMD = ROOT / "label.cmd"
LABEL_PS1 = ROOT / "scripts" / "label.ps1"
HELPER = ROOT / "engine" / "wearreport" / "tools" / "label_window.py"
WINDOWS_WORKFLOW = ROOT / ".github" / "workflows" / "windows.yml"
README = ROOT / "README.md"
WINDOWS = sys.platform == "win32"
WAIT_S = 900
UTC = datetime.UTC
MINUTE = datetime.timedelta(minutes=1)
JUDGE = "di-qwen3-vl-235b"
GUIDE_FLAGS = [
    ("--min-height", "46"),
    ("--min-persons", "1"),
    ("--timeout", "3600"),
    ("--n", str(spotcheck.MAX_N)),
]
WHERE = {"calgary": spotcheck.CALGARY, "london": spotcheck.LONDON}
MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"BM", b"RIFF", b"II*\x00", b"MM\x00*")

windows_only = pytest.mark.skipif(
    not WINDOWS, reason="the launcher is a Windows script; the windows-latest CI job runs it"
)


def _at(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text).replace(tzinfo=UTC)


def _lit(moment: datetime.datetime, city: str) -> bool:
    """Daylight as spotcheck decides it: anything but its "dark"."""
    return spotcheck.light_at(moment, WHERE[city]) != "dark"


def _in_london_window(moment: datetime.datetime) -> bool:
    """The London labelling window, 09:30-15:00 Europe/London (T-077)."""
    from zoneinfo import ZoneInfo

    local = moment.astimezone(ZoneInfo("Europe/London"))
    return 9 * 60 + 30 <= local.hour * 60 + local.minute < 15 * 60


def _expected_next(
    now: datetime.datetime, cities: tuple[str, ...]
) -> tuple[datetime.datetime, str]:
    """The first whole UTC minute at or after `now` with daylight in one of `cities`
    (Calgary first when both), by spotcheck's own light."""
    start = now.replace(second=0, microsecond=0)
    if start < now:
        start += MINUTE
    for i in range(48 * 60 + 1):
        moment = start + i * MINUTE
        for city in cities:
            if _lit(moment, city) and (city != "london" or _in_london_window(moment)):
                return moment, city
    raise AssertionError("no daylight within 48 hours")


# Fixed clocks, checked against spotcheck's own light before use.
CALGARY_DAY = _at("2026-06-21T19:00")  # 13:00 in Calgary, 20:00 in London
LONDON_ONLY = _at("2026-06-21T09:00")  # 03:00 in Calgary, 10:00 in London
BOTH_DARK = _at("2026-12-21T04:00")  # 21:00 in Calgary, 04:00 in London
CALGARY_DARK_LONDON_DAY = LONDON_ONLY


def test_fixed_clocks_are_what_they_claim() -> None:
    assert _lit(CALGARY_DAY, "calgary")
    assert _lit(LONDON_ONLY, "london") and not _lit(LONDON_ONLY, "calgary")
    assert not _lit(BOTH_DARK, "london") and not _lit(BOTH_DARK, "calgary")


# AC2: the city, on Linux with fixed clocks ------------------------------------------


def test_ac2_auto_picks_calgary_in_calgary_daylight() -> None:
    choice = label_window.choose("auto", CALGARY_DAY)
    assert choice.city == "calgary"


def test_ac2_auto_picks_london_when_only_london_is_lit() -> None:
    choice = label_window.choose("auto", LONDON_ONLY)
    assert choice.city == "london"


def test_ac2_auto_in_the_dark_gives_the_next_window_of_either_city() -> None:
    choice = label_window.choose("auto", BOTH_DARK)
    assert choice.city is None
    expected, city = _expected_next(BOTH_DARK, ("calgary", "london"))
    assert choice.next_window == expected
    assert choice.next_city == city


@pytest.mark.parametrize("seconds", [0, 1, 59])
def test_ac2_the_next_window_is_a_whole_minute_not_before_now(seconds: int) -> None:
    now = BOTH_DARK + datetime.timedelta(seconds=seconds)
    choice = label_window.choose("auto", now)
    assert choice.next_window is not None
    assert choice.next_window >= now
    assert choice.next_window.second == 0 and choice.next_window.microsecond == 0
    assert choice.next_window == _expected_next(now, ("calgary", "london"))[0]


@pytest.mark.parametrize("city", ["calgary", "london"])
def test_ac2_a_forced_city_is_used_in_its_daylight(city: str) -> None:
    now = CALGARY_DAY if city == "calgary" else LONDON_ONLY
    assert label_window.choose(city, now).city == city


def test_ac2_a_forced_city_in_its_dark_gives_its_own_next_window() -> None:
    choice = label_window.choose("calgary", CALGARY_DARK_LONDON_DAY)
    assert choice.city is None
    expected, city = _expected_next(CALGARY_DARK_LONDON_DAY, ("calgary",))
    assert (choice.next_window, choice.next_city) == (expected, "calgary")
    assert city == "calgary"


def test_ac2_daylight_matches_spotcheck_over_two_days() -> None:
    """The threshold is spotcheck's: daylight is anything it would not call dark."""
    start = _at("2026-03-29T00:00")  # a clock change in London
    for city in ("calgary", "london"):
        for i in range(0, 2 * 24 * 60, 7):
            moment = start + i * MINUTE
            assert label_window.in_daylight(city, moment) == _lit(moment, city), (city, moment)


def test_ac2_constants_are_spotchecks() -> None:
    assert label_window.CITIES["calgary"] == spotcheck.CALGARY
    assert label_window.CITIES["london"] == spotcheck.LONDON
    assert label_window.DARK_BELOW_DEG == spotcheck.LIGHT_TWILIGHT_DEG


def test_ac2_uses_the_engines_solar_code_not_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = ast.parse(HELPER.read_text(encoding="utf-8"))
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "solar_elevation" not in defined
    text = HELPER.read_text(encoding="utf-8")
    assert "acos" not in text and "asin" not in text
    seen: list[tuple[float, float]] = []

    def fake(moment: datetime.datetime, latitude: float, longitude: float) -> float:
        seen.append((latitude, longitude))
        return 10.0

    monkeypatch.setattr(pilot_heights, "solar_elevation", fake)
    assert label_window.choose("auto", BOTH_DARK).city == "calgary"
    assert seen and seen[0] == spotcheck.CALGARY


def test_ac2_command_line_prints_the_choice_as_json() -> None:
    def run(*args: str) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, "-m", "wearreport.tools.label_window", *args],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert isinstance(data, dict)
        return data

    assert run("city", "--source", "auto", "--now", "2026-06-21T19:00Z") == {"city": "calgary"}
    assert run("city", "--source", "london", "--now", "2026-06-21T09:00Z") == {"city": "london"}
    expected, city = _expected_next(BOTH_DARK, ("calgary", "london"))
    assert run("city", "--now", "2026-12-21T04:00Z") == {
        "city": None,
        "next_window": expected.strftime("%Y-%m-%dT%H:%MZ"),
        "next_city": city,
    }


@pytest.mark.parametrize(
    "args",
    [
        ["city", "--now", "yesterday"],
        ["city", "--now", "2026-06-21T19:00"],  # no time zone
        ["city", "--source", "austin"],
        ["city", "--now", "99999-01-01T00:00Z"],
    ],
)
def test_ac2_command_line_refuses_bad_input(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "wearreport.tools.label_window", *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


# AC5: counting the session's files -------------------------------------------------------


def _attribute_file(path: Path, crops: list[list[object]]) -> Path:
    record = {"min_height_px": 46, "crops_shown": len(crops) + 1, "crops": crops}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_ac5_count_reports_kept_and_judged_per_file_and_in_all(tmp_path: Path) -> None:
    one = _attribute_file(tmp_path / "a.json", [[50, "ynn", "ynn"], [60, "nnn", None]])
    two = _attribute_file(tmp_path / "b.json", [[70, "uuu", "yyy"]])
    counts = label_window.count([one, two])
    assert counts.kept == 3 and counts.judged == 2
    assert [(f.kept, f.judged) for f in counts.files] == [(2, 1), (1, 1)]
    result = subprocess.run(
        [sys.executable, "-m", "wearreport.tools.label_window", "count", str(one), str(two)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["kept"] == 3 and data["judged"] == 2
    assert [(f["kept"], f["judged"]) for f in data["files"]] == [(2, 1), (1, 1)]


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not json",
        b"[]",
        b'{"crops": 3}',
        b'{"crops": [[1, "ynn"]]}',
        b'{"crops": [[1, "ynn", 5]]}',
        b"\xff\xfe",
        b"[" * 100_000 + b"]" * 100_000,
        b'{"crops": []}' + b" " * (2 * 1024 * 1024),
    ],
    ids=["empty", "text", "list", "crops-int", "short", "model-int", "utf16", "deep", "huge"],
)
def test_ac5_count_refuses_a_malformed_file(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(label_window.LabelWindowError):
        label_window.count([path])
    result = subprocess.run(
        [sys.executable, "-m", "wearreport.tools.label_window", "count", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


# The launcher's files, as text --------------------------------------------------------


def test_ac1_label_cmd_starts_the_script_bypassing_policy_for_it_only() -> None:
    text = LABEL_CMD.read_text(encoding="utf-8")
    assert re.search(r"powershell(\.exe)?\s+.*-ExecutionPolicy\s+Bypass\s+.*-File", text, re.I)
    assert "scripts\\label.ps1" in text and "%~dp0" in text and "%*" in text
    assert re.search(r"^\s*pause\s*$", text, re.I | re.M)
    for text_ in (text, LABEL_PS1.read_text(encoding="utf-8")):
        assert "Set-ExecutionPolicy" not in text_


def test_launcher_files_are_ascii_english() -> None:
    for path in (LABEL_CMD, LABEL_PS1):
        path.read_bytes().decode("ascii")


def test_ac1_ac3_ac4_launcher_text_names_the_commands_and_settings() -> None:
    text = LABEL_PS1.read_text(encoding="utf-8")
    for needle in ("pull", "--ff-only", "sync", "--locked", "--no-install-package"):
        assert needle in text
    assert "llama-cpp-python" in text
    assert "TMPDIR" in text and "D:\\spotcheck-tmp" in text and "spotcheck-tmp" in text
    for flag in ("--attributes", "--view", "window", "--confirm-stop", "--out-dir", "--source"):
        assert flag in text
    assert "--judge" in text and "--judge-max-requests" in text
    assert JUDGE in text and JUDGE in judge_hosted.DEEPINFRA
    assert "DEEPINFRA_API_KEY" in text and "AsSecureString" in text


def test_ac6_launcher_never_handles_an_image_or_writes_a_log() -> None:
    text = LABEL_PS1.read_text(encoding="utf-8").lower()
    for needle in (
        ".png",
        ".jpg",
        "system.drawing",
        "setimage",
        "start-transcript",
        "out-file",
        "add-content",
        "invoke-item",
        "numbering",
    ):
        assert needle not in text, needle


def test_ac6_helper_imports_only_the_standard_library_and_the_engine() -> None:
    tree = ast.parse(HELPER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            assert top == "wearreport" or top in sys.stdlib_module_names, name


def test_ac7_windows_job_runs_this_file_and_stays_read_only() -> None:
    text = WINDOWS_WORKFLOW.read_text(encoding="utf-8")
    assert "engine/tests/acceptance/test_t_067.py" in text
    assert "runs-on: windows-latest" in text
    assert re.search(r"permissions:\s*\n\s*contents: read", text)
    assert "secrets." not in text and "pull_request_target" not in text


def test_readme_has_a_labelling_on_windows_paragraph() -> None:
    text = README.read_text(encoding="utf-8")
    assert "Labelling on Windows" in text
    section = text.split("Labelling on Windows", 1)[1][:2000]
    assert "label.cmd" in section


def test_ac7_public_guard_is_clean() -> None:
    for extra in ([], ["--history"]):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "public_guard.py"), *extra],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_ac6_privacy_guard_is_clean() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "privacy_guard.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# The launcher end to end, on Windows ------------------------------------------------

# The test reviewer and the judge's origin, in place of spotcheck's own module: it answers
# the `--judgements` file from the review's numbering file (never an image), and runs
# spotcheck's main with the fake judge on this machine. It logs its argument list, working
# directory and TMPDIR, never the key.
SHIM = r"""
import json, os, sys, tempfile, threading, time
from pathlib import Path
from wearreport.tools import spotcheck

argv = sys.argv[1:]
log = Path(os.environ["LABEL_TEST_LOG"])
with log.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({"argv": argv, "cwd": os.getcwd(),
                         "tmpdir": os.environ.get("TMPDIR"),
                         "keyed": bool(os.environ.get("DEEPINFRA_API_KEY"))}) + "\n")
number = len(log.read_text(encoding="utf-8").splitlines())
fail = {int(x) for x in os.environ.get("LABEL_TEST_FAIL", "").split(",") if x}
if number in fail:
    print("spotcheck: simulated failure", file=sys.stderr)
    raise SystemExit(1)

def answer(path):
    root = Path(tempfile.gettempdir())
    while True:
        for d in root.glob(spotcheck.TEMP_PREFIX + "*"):
            try:
                data = json.loads((d / spotcheck.NUMBERING_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            keys = sorted(data["template"], key=int)
            answers = {k: {"outer_layer": "y", "bare_legs": "n", "umbrella": "n"} for k in keys}
            answers[keys[0]] = "x"
            path.write_text(json.dumps(answers), encoding="utf-8")
            return
        time.sleep(0.2)

if "--judgements" in argv:
    target = Path(argv[argv.index("--judgements") + 1])
    threading.Thread(target=answer, args=(target,), daemon=True).start()
endpoint = os.environ.get("LABEL_TEST_JUDGE")
raise SystemExit(spotcheck.main(argv, judge_endpoint=endpoint, judge_sleep=lambda s: None))
"""

FAKE_CMD = (
    '@echo off\r\necho %~n0 %*>>"%LABEL_TEST_TOOLS_LOG%"\r\nexit /b %LABEL_TEST_TOOL_EXIT%\r\n'
)


@dataclass
class FakeJudge:
    fail: bool = False
    auth: list[str] = field(default_factory=list)
    server: ThreadingHTTPServer | None = None

    @property
    def endpoint(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def judge() -> Iterator[FakeJudge]:
    fake = FakeJudge()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            fake.auth.append(self.headers.get("Authorization", ""))
            if fake.fail:
                body = b'{"error": "bad request"}'
                self.send_response(400)
            else:
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "outer=yes legs=no umbrella=no",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 300, "completion_tokens": 12},
                    }
                ).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    fake.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=fake.server.serve_forever, daemon=True)
    thread.start()
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


@dataclass
class Session:
    code: int
    out: str
    calls: list[dict[str, object]]
    tools: list[str]
    files: list[Path]
    clipboard: str | None


@dataclass
class Launcher:
    tmp: Path
    judge: FakeJudge
    key: str = field(default_factory=lambda: "fake-test-key-" + secrets.token_hex(12))

    @property
    def temp_dir(self) -> Path:
        return self.tmp / "session-tmp"

    @property
    def out_dir(self) -> Path:
        return self.tmp / "out"

    @property
    def clipboard_file(self) -> Path:
        return self.tmp / "clipboard.txt"

    def run(
        self,
        *args: str,
        stdin: str = "",
        key_in_env: bool = True,
        fail: str = "",
        tool_exit: int = 0,
        dry_run: bool = True,
        now: str = "2026-06-21T19:00Z",
        via_cmd: bool = False,
    ) -> Session:
        shim = self.tmp / "shim.py"
        shim.write_text(SHIM, encoding="utf-8")
        log = self.tmp / "calls.jsonl"
        tools_log = self.tmp / "tools.log"
        for path in (log, tools_log):
            path.unlink(missing_ok=True)
        fakes = self.tmp / "fakebin"
        fakes.mkdir(exist_ok=True)
        for name in ("git", "uv"):
            (fakes / f"{name}.cmd").write_text(FAKE_CMD, encoding="ascii")
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in spotcheck.CI_VARIABLES and k not in ("DEEPINFRA_API_KEY", "TMPDIR")
        }
        env["PATH"] = str(fakes) + os.pathsep + env.get("PATH", "")
        env["LABEL_TEST_LOG"] = str(log)
        env["LABEL_TEST_TOOLS_LOG"] = str(tools_log)
        env["LABEL_TEST_TOOL_EXIT"] = str(tool_exit)
        env["LABEL_TEST_JUDGE"] = self.judge.endpoint
        env["LABEL_TEST_FAIL"] = fail
        if key_in_env:
            env["DEEPINFRA_API_KEY"] = self.key
        options = [
            "-TempDir",
            str(self.temp_dir),
            "-OutDir",
            str(self.out_dir),
            "-Python",
            sys.executable,
            "-Spotcheck",
            str(shim),
            "-ClipboardFile",
            str(self.clipboard_file),
            "-Now",
            now,
            *(["-DryRun"] if dry_run else []),
            *args,
        ]
        if via_cmd:
            command = ["cmd.exe", "/d", "/c", str(LABEL_CMD), *options]
        else:
            command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(LABEL_PS1),
                *options,
            ]
        result = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=self.tmp,
            timeout=WAIT_S,
            check=False,
        )
        out = result.stdout + result.stderr
        print(out)
        calls = (
            [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            if (log.exists())
            else []
        )
        tools = tools_log.read_text(encoding="utf-8").splitlines() if tools_log.exists() else []
        attributes = self.out_dir / spotcheck.ATTRIBUTES_DIR
        files = sorted(attributes.glob("*.json")) if attributes.is_dir() else []
        clipboard = (
            self.clipboard_file.read_text(encoding="utf-8")
            if self.clipboard_file.exists()
            else None
        )
        return Session(result.returncode, out, calls, tools, files, clipboard)

    def assert_no_key_and_no_leftovers(self, session: Session) -> None:
        assert self.key not in session.out
        for dirpath, _dirs, names in os.walk(self.tmp):
            for name in names:
                path = Path(dirpath, name)
                data = path.read_bytes()
                assert self.key.encode() not in data, path
                assert not data.startswith(MAGIC), path
        if self.temp_dir.exists():
            left = [p.name for p in self.temp_dir.iterdir() if not p.name.endswith(".lock")]
            assert left == []


@pytest.fixture
def launcher(tmp_path: Path, judge: FakeJudge) -> Launcher:
    return Launcher(tmp_path, judge)


def _summary(out: str) -> tuple[int, int, int]:
    match = re.search(r"(\d+) pass(?:es)?, (\d+) crops? kept, (\d+) crops? judged", out)
    assert match, out
    passes, kept, judged = match.groups()
    return int(passes), int(kept), int(judged)


def _kept(path: Path) -> int:
    crops = json.loads(path.read_text(encoding="utf-8"))["crops"]
    assert isinstance(crops, list)
    return len(crops)


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@windows_only
def test_ac3_ac4_ac5_two_passes_then_q(launcher: Launcher) -> None:
    session = launcher.run("-Target", "1000", stdin="\nq\n")
    assert session.code == 0
    assert len(session.calls) == 2
    assert "at your request" in session.out
    for call in session.calls:
        argv = call["argv"]
        assert isinstance(argv, list)
        assert "--attributes" in argv and "--dry-run" in argv and "--judgements" in argv
        for flag, value in GUIDE_FLAGS:
            assert _value(argv, flag) == value
        assert _value(argv, "--source") == "calgary"
        assert Path(_value(argv, "--out-dir")) == launcher.out_dir
        assert _value(argv, "--judge") == JUDGE
        assert 1 <= int(_value(argv, "--judge-max-requests")) <= spotcheck.MAX_JUDGE_REQUESTS
        assert launcher.key not in argv
        assert call["keyed"] is True
        assert Path(str(call["cwd"])).resolve() == ROOT.resolve()
        assert Path(str(call["tmpdir"])) == launcher.temp_dir
    assert len(session.files) == 2
    contents = [p.read_text(encoding="utf-8").strip() for p in session.files]
    assert session.clipboard is not None
    lines = session.clipboard.splitlines()
    assert sorted(lines) == sorted(contents)
    assert all(isinstance(json.loads(line), dict) for line in lines)
    assert session.clipboard.strip() == session.clipboard.rstrip("\r\n")
    for path in session.files:
        assert path.name in session.out
    kept = sum(_kept(p) for p in session.files)
    assert _summary(session.out) == (2, kept, kept)
    assert "Copied to the clipboard." in session.out
    assert launcher.judge.auth and all(a == f"Bearer {launcher.key}" for a in launcher.judge.auth)
    assert session.tools == [
        "git pull --ff-only",
        "uv sync --locked --no-install-package llama-cpp-python",
    ]
    launcher.assert_no_key_and_no_leftovers(session)


@windows_only
def test_ac3_stops_at_the_target(launcher: Launcher) -> None:
    session = launcher.run("-Target", "1", stdin="\n\n\n\n\n\n")
    assert session.code == 0
    assert len(session.calls) == 1
    assert "Target reached" in session.out
    assert len(session.files) == 1
    assert session.clipboard is not None
    assert session.clipboard.splitlines() == [session.files[0].read_text(encoding="utf-8").strip()]
    launcher.assert_no_key_and_no_leftovers(session)


@windows_only
def test_ac3_stops_after_max_passes(launcher: Launcher) -> None:
    session = launcher.run("-Target", "1000", "-MaxPasses", "2", stdin="\n\n\n\n")
    assert session.code == 0
    assert len(session.calls) == 2 and len(session.files) == 2
    assert _summary(session.out)[0] == 2
    launcher.assert_no_key_and_no_leftovers(session)


@windows_only
def test_ac3_one_failure_continues_two_in_a_row_stop(launcher: Launcher) -> None:
    session = launcher.run("-Target", "1000", fail="1,3,4", stdin="\n\n\n\n\n\n")
    assert session.code == 0
    assert len(session.calls) == 4
    assert "Pass 1 failed" in session.out
    assert "Pass 3 failed" in session.out and "Pass 4 failed" in session.out
    assert "Two passes in a row failed" in session.out
    assert len(session.files) == 1
    assert session.clipboard is not None
    assert session.clipboard.splitlines() == [session.files[0].read_text(encoding="utf-8").strip()]
    launcher.assert_no_key_and_no_leftovers(session)


@windows_only
def test_ac3_live_settings_use_the_window_and_two_failures_end_it(launcher: Launcher) -> None:
    """Without -DryRun, every pass asks for the window with --confirm-stop (the shim fails
    each pass, so no live sweep is ever made)."""
    session = launcher.run("-Target", "1000", fail="1,2,3", dry_run=False, stdin="\n\n\n")
    assert len(session.calls) == 2
    assert "Two passes in a row failed" in session.out
    for call in session.calls:
        argv = call["argv"]
        assert isinstance(argv, list)
        assert _value(argv, "--view") == "window" and "--confirm-stop" in argv
        assert "--dry-run" not in argv and "--judgements" not in argv
        for flag, value in GUIDE_FLAGS:
            assert _value(argv, flag) == value
    assert session.files == [] and session.clipboard is None
    launcher.assert_no_key_and_no_leftovers(session)


@windows_only
def test_ac4_asks_for_a_key_masked_and_does_not_save_it(launcher: Launcher) -> None:
    session = launcher.run("-Target", "1", key_in_env=False, stdin=f"{launcher.key}\nn\n\n\n")
    assert session.code == 0
    assert "DeepInfra" in session.out and "[y/N]" in session.out
    assert len(session.calls) == 1
    argv = session.calls[0]["argv"]
    assert isinstance(argv, list) and _value(argv, "--judge") == JUDGE
    assert session.calls[0]["keyed"] is True
    assert launcher.judge.auth and all(a == f"Bearer {launcher.key}" for a in launcher.judge.auth)
    launcher.assert_no_key_and_no_leftovers(session)


@windows_only
def test_ac4_an_empty_answer_means_no_judge(launcher: Launcher) -> None:
    session = launcher.run("-Target", "1", key_in_env=False, stdin="\n\n\n")
    assert session.code == 0
    argv = session.calls[0]["argv"]
    assert isinstance(argv, list)
    assert "--judge" not in argv and "--judge-max-requests" not in argv
    assert "[y/N]" not in session.out
    assert launcher.judge.auth == []
    assert _summary(session.out)[2] == 0


@windows_only
def test_ac4_no_judge_switch_skips_key_and_judge(launcher: Launcher) -> None:
    session = launcher.run("-Target", "1", "-NoJudge", stdin="\n\n")
    assert session.code == 0
    argv = session.calls[0]["argv"]
    assert isinstance(argv, list) and "--judge" not in argv
    assert "DeepInfra API key" not in session.out
    assert launcher.judge.auth == []
    launcher.assert_no_key_and_no_leftovers(session)


@windows_only
def test_ac4_a_judge_failure_does_not_stop_the_session(launcher: Launcher) -> None:
    launcher.judge.fail = True
    session = launcher.run("-Target", "1000", "-MaxPasses", "2", stdin="\n\n\n")
    assert session.code == 0
    assert len(session.calls) == 2 and len(session.files) == 2
    assert len(launcher.judge.auth) >= 2  # each pass opened its own judge
    passes, kept, judged = _summary(session.out)
    assert (passes, judged) == (2, 0) and kept > 0
    launcher.assert_no_key_and_no_leftovers(session)


@windows_only
def test_ac1_update_failures_are_one_line_each_and_temp_dir_is_made(launcher: Launcher) -> None:
    session = launcher.run("-Target", "1", tool_exit=1, stdin="\n\n")
    assert session.code == 0
    lines = session.out.splitlines()
    assert len([line for line in lines if "git pull failed" in line]) == 1
    assert len([line for line in lines if "uv sync failed" in line]) == 1
    assert len(session.calls) == 1 and len(session.files) == 1
    assert launcher.temp_dir.is_dir()
    assert Path(str(session.calls[0]["tmpdir"])) == launcher.temp_dir
    assert "TMPDIR" not in os.environ or os.environ["TMPDIR"] != str(launcher.temp_dir)


@windows_only
def test_ac2_london_when_calgary_is_dark(launcher: Launcher) -> None:
    session = launcher.run("-Target", "1", now="2026-06-21T09:00Z", stdin="\n\n")
    assert session.code == 0
    argv = session.calls[0]["argv"]
    assert isinstance(argv, list) and _value(argv, "--source") == "london"


@windows_only
def test_ac2_dark_everywhere_exits_with_the_next_window(launcher: Launcher) -> None:
    session = launcher.run(now="2026-12-21T04:00Z", stdin="\n\n")
    assert session.code != 0
    assert session.calls == []
    expected, _city = _expected_next(BOTH_DARK, ("calgary", "london"))
    assert expected.strftime("%Y-%m-%d %H:%M UTC") in session.out
    assert session.clipboard is None


@windows_only
def test_ac2_forced_source(launcher: Launcher) -> None:
    session = launcher.run("-Source", "london", "-Target", "1", now="2026-06-21T13:00Z")
    assert session.calls and all(
        _value(c["argv"], "--source") == "london"  # type: ignore[arg-type]
        for c in session.calls
    )


@windows_only
def test_ac1_label_cmd_runs_the_session_and_passes_its_arguments(launcher: Launcher) -> None:
    session = launcher.run("-Target", "1", via_cmd=True, stdin="\n\n\n\n")
    assert len(session.calls) == 1 and len(session.files) == 1
    assert "Copied to the clipboard." in session.out
    launcher.assert_no_key_and_no_leftovers(session)
