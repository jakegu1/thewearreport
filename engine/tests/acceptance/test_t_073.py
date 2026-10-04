"""Acceptance tests for T-073: the London labelling launcher (`label-london.cmd`: one London
pass, no hosted judge) and the written outer-layer codebook in `spotchecks/README.md`. The
task contract: do not edit.

On Linux these tests read the launcher, the guide, the README and the Windows workflow as
text. On Windows (the windows-latest CI job) `label-london.cmd` runs end to end with
`scripts/label.ps1`'s test switches, the same fakes as the T-067 tests: spotcheck's dry-run
pipeline (the licensed fixture photos, served on this machine), the `--judgements` file
reviewer answered by a test script, fake `git` and `uv` on PATH and a file in place of the
clipboard. Nothing reaches the network, and no image is opened or looked at: the test
reviewer reads only the review's numbering file.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from wearreport.tools import label_window, spotcheck

ROOT = Path(__file__).resolve().parents[3]
LABEL_LONDON_CMD = ROOT / "label-london.cmd"
LABEL_CMD = ROOT / "label.cmd"
LABEL_PS1 = ROOT / "scripts" / "label.ps1"
SPOTCHECKS_README = ROOT / "spotchecks" / "README.md"
README = ROOT / "README.md"
WINDOWS_WORKFLOW = ROOT / ".github" / "workflows" / "windows.yml"
WINDOWS = sys.platform == "win32"
WAIT_S = 900
UTC = datetime.UTC
MINUTE = datetime.timedelta(minutes=1)
GUIDE_FLAGS = [
    ("--min-height", "46"),
    ("--min-persons", "1"),
    ("--timeout", "3600"),
    ("--n", str(spotcheck.MAX_N)),
]
OUTER_QUESTION = "Outer layer (coat or jacket)?"
PAUSE_PROMPT = "Press any key to continue"

windows_only = pytest.mark.skipif(
    not WINDOWS, reason="the launcher is a Windows script; the windows-latest CI job runs it"
)


def _at(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text).replace(tzinfo=UTC)


def _lit(moment: datetime.datetime, where: tuple[float, float]) -> bool:
    return spotcheck.light_at(moment, where) != "dark"


# Both cities in daylight: a launcher that chose by itself would pick Calgary.
BOTH_LIT = _at("2026-06-21T19:00")  # 13:00 in Calgary, 20:00 in London
LONDON_DARK = _at("2026-12-21T04:00")  # 04:00 in London, 21:00 in Calgary


def _london_next_window(now: datetime.datetime) -> datetime.datetime:
    start = now.replace(second=0, microsecond=0)
    if start < now:
        start += MINUTE
    for i in range(48 * 60 + 1):
        moment = start + i * MINUTE
        if _lit(moment, spotcheck.LONDON):
            return moment
    raise AssertionError("no London daylight within 48 hours")


def test_fixed_clocks_are_what_they_claim() -> None:
    assert _lit(BOTH_LIT, spotcheck.LONDON) and _lit(BOTH_LIT, spotcheck.CALGARY)
    assert label_window.choose("auto", BOTH_LIT).city == "calgary"
    assert not _lit(LONDON_DARK, spotcheck.LONDON)


# AC1: the launcher, as text ---------------------------------------------------------


def _commands(text: str) -> list[str]:
    """The batch file's command lines: no blank lines, no `rem` comments."""
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line and not re.match(r"(?i)^rem(\s|$)", line)]


def test_ac1_label_london_cmd_is_ascii_english() -> None:
    LABEL_LONDON_CMD.read_bytes().decode("ascii")


def test_ac1_starts_label_ps1_like_label_cmd_with_the_london_settings_first() -> None:
    text = LABEL_LONDON_CMD.read_text(encoding="ascii")
    commands = _commands(text)
    starts = [c for c in commands if re.match(r"(?i)^powershell(\.exe)?\s", c)]
    assert len(starts) == 1, commands
    start = starts[0]
    match = re.fullmatch(
        r"(?i)powershell(?:\.exe)?\s+-NoProfile\s+-ExecutionPolicy\s+Bypass\s+-File\s+"
        r'"%~dp0scripts\\label\.ps1"\s+(?P<args>.*)',
        start,
    )
    assert match, start
    args = match.group("args").split()
    assert args == ["-Source", "london", "-MaxPasses", "1", "-NoJudge", "%*"], args
    assert any(re.fullmatch(r'(?i)cd\s+/d\s+"%~dp0"', c) for c in commands), commands
    assert "Set-ExecutionPolicy" not in text


def test_ac1_one_pause_and_the_scripts_exit_code() -> None:
    commands = _commands(LABEL_LONDON_CMD.read_text(encoding="ascii"))
    assert [c for c in commands if re.fullmatch(r"(?i)pause", c)] == ["pause"]
    saves = [c for c in commands if re.fullmatch(r'(?i)set\s+"(\w+)=%ERRORLEVEL%"', c)]
    assert len(saves) == 1, commands
    name = re.fullmatch(r'(?i)set\s+"(\w+)=%ERRORLEVEL%"', saves[0])
    assert name
    assert commands[-1].lower() == f"exit /b %{name.group(1)}%".lower(), commands
    index = commands.index
    assert index(saves[0]) < index("pause") < len(commands) - 1


# AC3: the codebook --------------------------------------------------------------------


def _attribute_section() -> str:
    text = SPOTCHECKS_README.read_text(encoding="utf-8")
    start = text.index("## Attribute session (`--attributes`)")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)]


def _codebook() -> str:
    """The codebook, on one line (phrases may wrap anywhere): from "outermost garment" to
    the end of the "Questions and keys" subsection, which it must sit in."""
    section = _attribute_section()
    questions = section.index("### Questions and keys")
    end = section.find("\n### ", questions + 1)
    sub = section[questions : end if end != -1 else len(section)]
    assert OUTER_QUESTION in sub
    start = sub.lower().index("outermost garment")
    assert start > sub.index(OUTER_QUESTION)
    return " ".join(sub[start:].split())


def test_ac3_codebook_sits_under_the_attribute_questions() -> None:
    assert _codebook()


@pytest.mark.parametrize(
    "phrase",
    [
        "outermost garment",
        "with sleeves",
        "opens at the front",
        "coats",
        "down or puffer jackets",
        "rain jackets",
        "blazers and suit jackets",
        "denim or leather jackets",
        "fleece jackets",
        "T-shirt",
        "shirt",
        "sweater",
        "cardigan",
        "hoodie",
        "with a hood or a zip",
        "sleeveless vest",
        "outermost layer",
        "cannot be told",
        "do not guess",
        "shorts or a short skirt",
        "open umbrella",
    ],
)
def test_ac3_codebook_states_the_rule(phrase: str) -> None:
    assert phrase.lower() in _codebook().lower(), phrase


def test_ac3_codebook_gives_y_n_and_u_their_garments() -> None:
    text = _codebook()
    y, n, u = (text.index(f"`{key}`") for key in ("y", "n", "u"))
    assert y < n < u, text
    yes, no, unsure = text[y:n].lower(), text[n:u].lower(), text[u:].lower()
    for garment in ("coat", "puffer", "rain jacket", "blazer", "leather", "fleece"):
        assert garment in yes, garment
    for garment in ("t-shirt", "shirt", "sweater", "cardigan", "hoodie", "sleeveless vest"):
        assert garment in no, garment
        if garment != "shirt":
            assert garment not in yes, garment
    assert "cannot be told" in unsure and "do not guess" in unsure


def test_ac3_the_question_shown_in_the_window_is_unchanged() -> None:
    questions = dict(spotcheck.ATTRIBUTE_QUESTIONS)
    assert questions["outer_layer"] == OUTER_QUESTION
    assert questions["bare_legs"] == "Bare legs (shorts or short skirt)?"
    assert questions["umbrella"] == "Holding an open umbrella?"


# AC4: the README ----------------------------------------------------------------------


def _labelling_paragraph() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index("Labelling on Windows")
    end = text.find("\n#", start)
    return " ".join(text[start : end if end != -1 else len(text)].split())


def test_ac4_readme_documents_both_launchers() -> None:
    paragraph = _labelling_paragraph()
    assert "label-london.cmd" in paragraph
    assert re.search(r"(?<![-\w])label\.cmd", paragraph)
    tail = paragraph[paragraph.index("label-london.cmd") :]
    assert "London" in tail
    assert re.search(r"one pass", tail, re.I)
    assert re.search(r"(no|without the hosted|without the) judge|-NoJudge", tail, re.I)


# AC5: unchanged, and the Windows job ---------------------------------------------------


def test_ac5_label_cmd_keeps_its_defaults() -> None:
    text = LABEL_CMD.read_text(encoding="utf-8")
    start = [c for c in _commands(text) if re.match(r"(?i)^powershell", c)]
    assert len(start) == 1 and start[0].rstrip().endswith('label.ps1" %*'), start
    script = LABEL_PS1.read_text(encoding="utf-8")
    assert "[string] $Source = 'auto'" in script
    assert "[int] $MaxPasses = 6" in script


def test_ac5_windows_job_runs_this_file_and_stays_read_only() -> None:
    text = WINDOWS_WORKFLOW.read_text(encoding="utf-8")
    assert "engine/tests/acceptance/test_t_073.py" in text
    assert "engine/tests/acceptance/test_t_067.py" in text
    assert "runs-on: windows-latest" in text
    assert re.search(r"permissions:\s*\n\s*contents: read", text)
    assert "secrets." not in text and "pull_request_target" not in text


def test_ac5_public_guard_is_clean() -> None:
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


# AC2: label-london.cmd end to end, on Windows ---------------------------------------------

# The test reviewer, in place of spotcheck's own module: it answers the `--judgements` file
# from the review's numbering file (never an image) and runs spotcheck's main. It logs its
# argument list and whether a key was in its environment, never the key.
SHIM = r"""
import json, os, sys, tempfile, threading, time
from pathlib import Path
from wearreport.tools import spotcheck

argv = sys.argv[1:]
log = Path(os.environ["LABEL_TEST_LOG"])
with log.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({"argv": argv, "keyed": bool(os.environ.get("DEEPINFRA_API_KEY"))}) + "\n")

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
            path.write_text(json.dumps(answers), encoding="utf-8")
            return
        time.sleep(0.2)

if "--judgements" in argv:
    target = Path(argv[argv.index("--judgements") + 1])
    threading.Thread(target=answer, args=(target,), daemon=True).start()
raise SystemExit(spotcheck.main(argv))
"""

FAKE_CMD = (
    '@echo off\r\necho %~n0 %*>>"%LABEL_TEST_TOOLS_LOG%"\r\nexit /b %LABEL_TEST_TOOL_EXIT%\r\n'
)


@dataclass
class Session:
    code: int
    out: str
    calls: list[dict[str, object]]
    files: list[Path]
    clipboard: str | None


def _run(tmp: Path, now: str, stdin: str = "\n\n\n\n") -> Session:
    """label-london.cmd through cmd.exe, as a double-click starts it, with a key in the
    environment (which -NoJudge must ignore) and the test switches after it."""
    shim = tmp / "shim.py"
    shim.write_text(SHIM, encoding="utf-8")
    log = tmp / "calls.jsonl"
    fakes = tmp / "fakebin"
    fakes.mkdir(exist_ok=True)
    for name in ("git", "uv"):
        (fakes / f"{name}.cmd").write_text(FAKE_CMD, encoding="ascii")
    out_dir = tmp / "out"
    clipboard = tmp / "clipboard.txt"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in spotcheck.CI_VARIABLES and k not in ("DEEPINFRA_API_KEY", "TMPDIR")
    }
    env["PATH"] = str(fakes) + os.pathsep + env.get("PATH", "")
    env["LABEL_TEST_LOG"] = str(log)
    env["LABEL_TEST_TOOLS_LOG"] = str(tmp / "tools.log")
    env["LABEL_TEST_TOOL_EXIT"] = "0"
    env["DEEPINFRA_API_KEY"] = "fake-test-key-not-used"
    command = [
        "cmd.exe",
        "/d",
        "/c",
        str(LABEL_LONDON_CMD),
        "-TempDir",
        str(tmp / "session-tmp"),
        "-OutDir",
        str(out_dir),
        "-Python",
        sys.executable,
        "-Spotcheck",
        str(shim),
        "-ClipboardFile",
        str(clipboard),
        "-Now",
        now,
        "-DryRun",
    ]
    result = subprocess.run(
        command,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=tmp,
        timeout=WAIT_S,
        check=False,
    )
    out = result.stdout + result.stderr
    print(out)
    calls = (
        [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if log.exists()
        else []
    )
    for call in calls:
        print("spotcheck arguments:", json.dumps(call["argv"]))
    attributes = out_dir / spotcheck.ATTRIBUTES_DIR
    files = sorted(attributes.glob("*.json")) if attributes.is_dir() else []
    text = clipboard.read_text(encoding="utf-8") if clipboard.exists() else None
    return Session(result.returncode, out, calls, files, text)


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@windows_only
def test_ac2_one_london_pass_without_the_judge(tmp_path: Path) -> None:
    session = _run(tmp_path, now="2026-06-21T19:00Z")
    assert session.code == 0
    assert len(session.calls) == 1
    argv = session.calls[0]["argv"]
    assert isinstance(argv, list)
    assert _value(argv, "--source") == "london"
    assert "--attributes" in argv and "--dry-run" in argv
    for flag, value in GUIDE_FLAGS:
        assert _value(argv, flag) == value
    assert "--judge" not in argv and "--judge-max-requests" not in argv
    assert "DeepInfra API key" not in session.out
    assert "City: London." in session.out
    assert "Reached the maximum of 1 passes." in session.out
    assert "Next pass in" not in session.out
    assert len(session.files) == 1
    assert session.clipboard is not None
    assert session.clipboard.splitlines() == [session.files[0].read_text(encoding="utf-8").strip()]
    assert isinstance(json.loads(session.clipboard), dict)
    assert "Copied to the clipboard." in session.out
    assert session.out.count(PAUSE_PROMPT) == 1


@windows_only
def test_ac2_in_the_dark_exits_with_londons_next_window(tmp_path: Path) -> None:
    session = _run(tmp_path, now="2026-12-21T04:00Z")
    assert session.code == 1
    assert session.calls == []
    expected = _london_next_window(LONDON_DARK)
    opens = expected.strftime("%Y-%m-%d %H:%M UTC")
    assert f"No daylight in London now. The next window opens at {opens} (London)." in session.out
    assert session.clipboard is None
    assert session.out.count(PAUSE_PROMPT) == 1
