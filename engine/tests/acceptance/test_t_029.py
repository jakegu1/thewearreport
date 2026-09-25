"""Acceptance tests for T-029 (spot-check judge bake-off: a licensed gold set and a local
open vision model). The task contract: do not edit.

Every image in these tests is synthetic, a committed control crop, or a crop of an
openly licensed gold-set source image. No test reads a camera frame.

Tests that need the judge weights (scripts/fetch_judge_model.sh) or the gold-set source
images (scripts/fetch_goldset.sh) skip only when those files are missing and
WEARREPORT_REQUIRE_JUDGE is unset; the judge workflow sets it, so there a missing file
fails. CI does not download the weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import socket
import stat
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, get_args

import numpy as np
import numpy.typing as npt
import pytest

import license_check
import privacy_guard
from wearreport import detect
from wearreport._cv import cv2
from wearreport.tools import goldset, judge, spotcheck

ROOT = Path(__file__).resolve().parents[3]
REQUIRE_JUDGE = "WEARREPORT_REQUIRE_JUDGE"
FETCH_GOLDSET = ROOT / "scripts" / "fetch_goldset.sh"
FETCH_JUDGE = ROOT / "scripts" / "fetch_judge_model.sh"
CONTROLS = ROOT / "fixtures" / "goldset" / "controls"
JUDGE_WORKFLOW = ROOT / ".github" / "workflows" / "judge.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
NEW_MODULES = ("engine/wearreport/tools/judge.py", "engine/wearreport/tools/goldset.py")
ALLOWED_LICENCE = re.compile(r"(CC0 1\.0|Public domain|CC BY \d\.\d)", re.IGNORECASE)
SAFE_CURL = "curl --proto '=https' --tlsv1.2"
SHA256 = re.compile(r"[0-9a-f]{64}")


def _require(what: str) -> None:
    if os.environ.get(REQUIRE_JUDGE):
        pytest.fail(f"{what} is missing and {REQUIRE_JUDGE} is set")
    pytest.skip(f"{what} is missing")


def _chosen() -> judge.Candidate:
    return judge.CANDIDATES[judge.CHOSEN]


def _weights_present(candidate: judge.Candidate) -> bool:
    return all(
        (judge.MODEL_DIR / name).is_file() for name in (candidate.model_file, candidate.mmproj_file)
    )


@pytest.fixture(scope="module")
def chosen_judge() -> Iterator[judge.Judge]:
    candidate = _chosen()
    if not _weights_present(candidate):
        _require(f"the weights of {candidate.name}")
    jd = judge.Judge(candidate)
    try:
        yield jd
    finally:
        jd.close()


@pytest.fixture(scope="module")
def manifest() -> goldset.Manifest:
    return goldset.load_manifest()


def _sources_present(manifest: goldset.Manifest) -> bool:
    return all((goldset.SOURCE_DIR / s.id).is_file() for s in manifest.sources.values())


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every way out to the network raises."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"network access attempted: {args!r}")

    for name in ("connect", "connect_ex", "sendto", "bind"):
        monkeypatch.setattr(socket.socket, name, refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


def _synthetic_frame(height: int = 288, width: int = 352, seed: int = 1) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    frame: npt.NDArray[np.uint8] = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (150, 80), (180, 200), (40, 40, 200), -1)  # a figure-sized block
    return frame


# AC1: the gold set ----------------------------------------------------------------------


def test_ac1_gold_set_size_and_classes(manifest: goldset.Manifest) -> None:
    assert len(manifest.items) >= 400
    assert goldset.LABELS == ("person", "in_vehicle", "not_person")
    assert set(get_args(goldset.Label)) == set(goldset.LABELS)
    labels = [item.label for item in manifest.items]
    assert set(labels) == set(goldset.LABELS)
    assert len({item.id for item in manifest.items}) == len(manifest.items)


def test_ac1_at_least_a_quarter_are_hard_negatives(manifest: goldset.Manifest) -> None:
    kinds = {"pole", "bin", "sign", "bollard", "shadow", "depiction"}
    assert set(goldset.HARD_NEGATIVE_KINDS) == kinds
    hard = [item for item in manifest.items if item.kind in goldset.HARD_NEGATIVE_KINDS]
    assert all(item.label == "not_person" for item in hard)
    assert all(
        item.hard_negative == (item.kind in goldset.HARD_NEGATIVE_KINDS) for item in manifest.items
    )
    assert len(hard) >= 0.25 * len(manifest.items)
    assert any(item.kind == "depiction" for item in hard)


def test_ac1_every_source_is_openly_licensed_and_attributed(
    manifest: goldset.Manifest,
) -> None:
    used = {item.source for item in manifest.items}
    assert used == set(manifest.sources)
    for source in manifest.sources.values():
        assert source.url.startswith("https://"), source.id
        assert source.page.startswith("https://"), source.id
        assert source.author.strip(), source.id
        assert ALLOWED_LICENCE.fullmatch(source.license), source.license
        assert not re.search(r"\b(SA|NC|ND)\b", source.license.upper()), source.license
        assert SHA256.fullmatch(source.sha256), source.id
        assert source.width > 0 and source.height > 0
    raw = json.loads((goldset.MANIFEST_PATH).read_text())
    assert ALLOWED_LICENCE.fullmatch(raw["annotations"]["license"])
    assert raw["annotations"]["url"].startswith("https://")


def test_ac1_no_camera_captures_among_the_sources(manifest: goldset.Manifest) -> None:
    for source in manifest.sources.values():
        for url in (source.url, source.page):
            assert not re.search(r"tfl|jamcam|traffic.?cam", url, re.IGNORECASE), url


def test_ac1_boxes_lie_inside_their_source(manifest: goldset.Manifest) -> None:
    for item in manifest.items:
        source = manifest.sources[item.source]
        x1, y1, x2, y2 = item.box
        assert all(type(v) is int for v in item.box), item.id
        assert 0 <= x1 < x2 <= source.width and 0 <= y1 < y2 <= source.height, item.id


def test_ac1_degradation_is_within_tfl_conditions_and_seeded(
    manifest: goldset.Manifest,
) -> None:
    assert goldset.CROP_MARGIN == spotcheck.CROP_MARGIN == 0.5
    assert (goldset.MIN_HEIGHT_PX, goldset.MAX_HEIGHT_PX) == (15, 80)
    assert (goldset.MIN_JPEG_QUALITY, goldset.MAX_JPEG_QUALITY) == (35, 60)
    for item in manifest.items:
        assert 15 <= item.height_px <= 80, item.id
        assert 35 <= item.jpeg_quality <= 60, item.id
        assert goldset.degradation_params(manifest.seed, item.id) == (
            item.height_px,
            item.jpeg_quality,
        ), item.id
    heights = {item.height_px for item in manifest.items}
    assert min(heights) <= 25 and max(heights) >= 70  # the whole range is used


def test_ac1_degrade_is_deterministic_and_scales_the_box() -> None:
    frame = _synthetic_frame(1000, 800)
    box = (300, 200, 400, 500)  # 300 px high
    first = goldset.degrade(frame, box, 40, 50)
    second = goldset.degrade(frame, box, 40, 50)
    assert first.jpeg == second.jpeg
    assert first.jpeg[:3] == b"\xff\xd8\xff"
    x1, y1, x2, y2 = first.box
    assert abs((y2 - y1) - 40) <= 1
    height, width = first.frame.shape[:2]
    # the 50% margin: the crop is about twice the box on each axis
    assert abs(height - 80) <= 2 and abs(width - 2 * (x2 - x1)) <= 2
    ok, expected = cv2.imencode(".jpg", first.small, [cv2.IMWRITE_JPEG_QUALITY, 50])
    assert ok and expected.tobytes() == first.jpeg
    decoded = cv2.imdecode(np.frombuffer(first.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and np.array_equal(decoded, first.frame)


@pytest.mark.parametrize("seed", range(20))
def test_ac1_crop_bounds_and_rendering_match_the_spotcheck_tool(seed: int) -> None:
    rng = np.random.default_rng(seed)
    height, width = int(rng.integers(20, 300)), int(rng.integers(20, 400))
    frame: npt.NDArray[np.uint8] = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    x1, y1 = float(rng.uniform(0, width - 5)), float(rng.uniform(0, height - 5))
    box = (x1, y1, float(rng.uniform(x1 + 1, width)), float(rng.uniform(y1 + 1, height)))
    assert goldset.crop_bounds(box, width, height) == spotcheck.crop_bounds(box, width, height)
    number = int(rng.integers(1, 500))
    ours = goldset.render_crop(frame, box, number)
    theirs = spotcheck.render_crop(frame, detect.Detection("person", 0.9, box), number)
    assert ours.dtype == theirs.dtype and np.array_equal(ours, theirs)


def test_ac1_no_face_detector_anywhere() -> None:
    pattern = re.compile(
        r"CascadeClassifier|haarcascade|FaceDetectorYN|face_recognition|mediapipe|dlib"
        r"|detect_?faces?|face_?detect",
        re.IGNORECASE,
    )
    paths = [ROOT / p for p in NEW_MODULES] + [FETCH_GOLDSET, FETCH_JUDGE, JUDGE_WORKFLOW]
    for path in paths:
        assert not pattern.search(path.read_text()), path
    lock = (ROOT / "uv.lock").read_text().lower()
    for name in ("mediapipe", "dlib", "face-recognition", "insightface", "retinaface"):
        assert not re.search(rf'^name = "{name}"$', lock, flags=re.MULTILINE), name


def test_ac1_source_images_are_ignored_and_never_committed() -> None:
    git = shutil.which("git")
    assert git is not None
    rel = goldset.SOURCE_DIR.relative_to(ROOT).as_posix()
    assert f"{rel}/" in (ROOT / ".gitignore").read_text().splitlines()
    tracked = subprocess.run(
        [git, "ls-files", "fixtures/goldset", rel],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "fixtures/goldset/manifest.json" in tracked
    others = [p for p in tracked if p != "fixtures/goldset/manifest.json"]
    assert others and all(p.startswith("fixtures/goldset/controls/") for p in others)
    ignored = subprocess.run([git, "check-ignore", "-q", f"{rel}/x"], cwd=ROOT, check=False)
    assert ignored.returncode == 0


def test_ac1_gold_crops_render_from_the_downloaded_sources(
    manifest: goldset.Manifest,
) -> None:
    if not _sources_present(manifest):
        _require("the gold-set source images")
    count = 0
    for item, image in goldset.iter_gold(manifest):
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, item.id
        count += 1
    assert count == len(manifest.items)


# fetch_goldset.sh ----------------------------------------------------------------------


def _fake_curl(tmp_path: Path, body: str) -> dict[str, str]:
    """An environment whose `curl` logs its arguments and runs `body` into its -o file.

    `body` is shell code; `$url` holds the last argument.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$CURL_LOG"\n'
        'out=""; prev=""; url=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; url="$a"; done\n'
        f'{body} > "$out"\n'
    )
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_")}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["CURL_LOG"] = str(tmp_path / "curl.log")
    return env


def _sh(script: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    sh = shutil.which("sh")
    assert sh is not None
    return subprocess.run(
        [sh, str(script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _mini_manifest(tmp_path: Path, body: bytes) -> Path:
    data = json.loads(goldset.MANIFEST_PATH.read_text())
    source = dict(data["sources"][0])
    source["sha256"] = hashlib.sha256(body).hexdigest()
    data["sources"] = [source]
    data["items"] = [i for i in data["items"] if i["source"] == source["id"]]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path


def test_ac1_fetch_goldset_installs_verified_files(tmp_path: Path) -> None:
    body = b"\xff\xd8\xff pretend jpeg"
    manifest = _mini_manifest(tmp_path, body)
    (tmp_path / "body").write_bytes(body)
    env = _fake_curl(tmp_path, f"cat '{tmp_path / 'body'}'")
    dest = tmp_path / "gold"
    dest.mkdir()
    (dest / ".stale.part.abc123").write_bytes(b"left by a killed run")
    proc = _sh(FETCH_GOLDSET, env, "--manifest", str(manifest), "--dest", str(dest))
    assert proc.returncode == 0, proc.stderr
    source_id = json.loads(manifest.read_text())["sources"][0]["id"]
    assert sorted(p.name for p in dest.iterdir()) == [source_id]
    assert (dest / source_id).read_bytes() == body
    assert f"{SAFE_CURL}" in "curl " + (tmp_path / "curl.log").read_text().replace(
        "--proto =https", "--proto '=https'"
    )
    (tmp_path / "curl.log").unlink()
    assert _sh(FETCH_GOLDSET, env, "--manifest", str(manifest), "--dest", str(dest)).returncode == 0
    assert not (tmp_path / "curl.log").exists()  # verified files are not downloaded again


def test_ac1_fetch_goldset_fails_closed(tmp_path: Path) -> None:
    manifest = _mini_manifest(tmp_path, b"the real bytes")
    env = _fake_curl(tmp_path, "printf 'tampered bytes'")
    dest = tmp_path / "gold"
    dest.mkdir()
    source_id = json.loads(manifest.read_text())["sources"][0]["id"]
    (dest / source_id).write_bytes(b"corrupted earlier download")
    proc = _sh(FETCH_GOLDSET, env, "--manifest", str(manifest), "--dest", str(dest))
    assert proc.returncode != 0
    assert list(dest.iterdir()) == []  # neither the bad file nor a part file is left


def test_ac1_fetch_goldset_uses_safe_curl_and_no_tokens() -> None:
    script = FETCH_GOLDSET.read_text()
    assert SAFE_CURL in script and "--max-time" in script and "--retry" in script
    assert not re.search(r"token|authorization|api[_-]?key", script, re.IGNORECASE)


# AC2: candidates ----------------------------------------------------------------------


def test_ac2_three_to_five_permissively_licensed_candidates() -> None:
    candidates = list(judge.CANDIDATES.values())
    assert 3 <= len(candidates) <= 5
    assert all(name == c.name for name, c in judge.CANDIDATES.items())
    for c in candidates:
        assert c.licence in ("Apache-2.0", "MIT"), c.name
        assert c.family and c.params_b > 0, c.name
        assert re.fullmatch(r"[0-9a-f]{40}", c.revision), c.name  # a pinned commit
        assert SHA256.fullmatch(c.model_sha256) and SHA256.fullmatch(c.mmproj_sha256), c.name
        assert c.model_bytes > 0 and c.mmproj_bytes > 0, c.name
    assert any(re.match(r"(Qwen.*VL|Qwen3\.5|GLM.*V)", c.family) for c in candidates)
    assert any(c.params_b <= 3.0 for c in candidates)


def test_ac2_candidate_urls_are_pinned_hugging_face_downloads() -> None:
    for c in judge.CANDIDATES.values():
        for name in (c.model_file, c.mmproj_file):
            url = judge.download_url(c, name)
            assert url == f"https://huggingface.co/{c.repo}/resolve/{c.revision}/{name}"


# AC3: local and private ---------------------------------------------------------------


def test_ac3_telemetry_settings_are_in_the_environment() -> None:
    assert judge.TELEMETRY_ENV
    for key, value in judge.TELEMETRY_ENV.items():
        assert os.environ[key] == value, key
    assert judge.TELEMETRY_ENV.get("HF_HUB_DISABLE_TELEMETRY") == "1"


def _python(code: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], list[Path]]:
    """Run `code` with an empty HOME and TMPDIR and without the telemetry settings; return
    the result and every file the run left in those two directories."""
    home, tmp = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    tmp.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in judge.TELEMETRY_ENV}
    env.update(HOME=str(home), TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE="1")
    env = {k: v for k, v in env.items() if not k.startswith("XDG_")}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return proc, sorted(p for d in (home, tmp) for p in d.rglob("*"))


def test_ac3_importing_the_runtime_leaves_no_files(tmp_path: Path) -> None:
    code = "from wearreport.tools import judge\njudge.load_runtime()\nprint('ok')\n"
    proc, left = _python(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert left == []


def test_ac3_inference_leaves_no_files(tmp_path: Path) -> None:
    if not _weights_present(_chosen()):
        _require("the judge weights")
    code = (
        "import numpy as np\n"
        "from wearreport.tools import judge\n"
        "jd = judge.Judge(judge.CANDIDATES[judge.CHOSEN])\n"
        "print(jd.classify(np.full((120, 60, 3), 90, np.uint8)))\n"
        "jd.close()\n"
    )
    proc, left = _python(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] in get_args(judge.Answer)
    assert left == []


def test_ac3_the_file_check_itself_sees_files(tmp_path: Path) -> None:
    code = (
        "import pathlib\n"
        "(pathlib.Path.home() / '.cache').mkdir()\n"
        "(pathlib.Path.home() / '.cache' / 'planted').write_text('x')\n"
        "from wearreport.tools import judge\n"
    )
    proc, left = _python(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert any(p.name == "planted" for p in left)


def test_ac3_import_refuses_a_runtime_loaded_without_the_settings(tmp_path: Path) -> None:
    proc, _ = _python("import llama_cpp\nimport wearreport.tools.judge\n", tmp_path)
    assert proc.returncode != 0
    assert "ImportError" in proc.stderr and "telemetry" in proc.stderr


def test_ac3_judge_refuses_weights_that_are_not_pinned(tmp_path: Path) -> None:
    candidate = _chosen()
    (tmp_path / candidate.model_file).write_bytes(b"GGUF not really a model")
    (tmp_path / candidate.mmproj_file).write_bytes(b"GGUF not really a projector")
    with pytest.raises(judge.JudgeError):
        judge.Judge(candidate, model_dir=tmp_path)
    with pytest.raises(judge.JudgeError):
        judge.Judge(candidate, model_dir=tmp_path / "missing")


def test_ac3_classifies_with_every_socket_blocked(chosen_judge: judge.Judge, offline: None) -> None:
    frame = _synthetic_frame()
    image = goldset.render_crop(frame, (150.0, 80.0, 181.0, 201.0), 1)
    assert chosen_judge.classify(image) in get_args(judge.Answer)
    with pytest.raises(AssertionError):
        socket.create_connection(("192.0.2.1", 80))  # the block is in force


def test_ac3_runs_on_cpu_within_16_gb(chosen_judge: judge.Judge) -> None:
    assert judge.DEFAULT_THREADS == 4
    chosen_judge.classify(np.full((160, 80, 3), 120, np.uint8))
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert peak_kib < 16 * 1024 * 1024


def test_ac3_fetch_judge_model_pins_the_chosen_weights() -> None:
    script = FETCH_JUDGE.read_text()
    candidate = _chosen()
    assert SAFE_CURL in script and "--max-time" in script and "--retry" in script
    for name, digest in (
        (candidate.model_file, candidate.model_sha256),
        (candidate.mmproj_file, candidate.mmproj_sha256),
    ):
        assert name in script and digest in script, name
    assert candidate.revision in script and candidate.repo in script
    assert not re.search(r"authorization|HF_TOKEN|api[_-]?key", script, re.IGNORECASE)


def test_ac3_fetch_judge_model_fails_closed(tmp_path: Path) -> None:
    env = _fake_curl(tmp_path, "printf 'not a model'")
    dest = tmp_path / "judge"
    dest.mkdir()
    (dest / _chosen().model_file).write_bytes(b"corrupted")
    (dest / ".x.part.abc123").write_bytes(b"stale")
    proc = _sh(FETCH_JUDGE, env, "--dest", str(dest))
    assert proc.returncode != 0
    assert list(dest.iterdir()) == []
    log = (tmp_path / "curl.log").read_text()
    assert "--proto =https --tlsv1.2" in log and "huggingface.co" in log


def test_ac3_weights_are_ignored_and_never_committed() -> None:
    rel = judge.MODEL_DIR.relative_to(ROOT).as_posix()
    git = shutil.which("git")
    assert git is not None
    ignored = subprocess.run(
        [git, "check-ignore", "-q", f"{rel}/{_chosen().model_file}"], cwd=ROOT, check=False
    )
    assert ignored.returncode == 0
    tracked = subprocess.run(
        [git, "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert [p for p in tracked if p.endswith(".gguf")] == []


# AC4: the bake-off harness ------------------------------------------------------------


def test_ac4_prompt_is_fixed_and_names_every_answer() -> None:
    assert isinstance(judge.PROMPT, str) and judge.PROMPT.strip()
    assert set(get_args(judge.Answer)) == {"person", "in_vehicle", "not_person", "unsure"}
    for word in judge.ANSWER_WORDS:
        assert word in judge.PROMPT


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("person", "person"),
        (" Person.\n", "person"),
        ("vehicle", "in_vehicle"),
        ("other", "not_person"),
        ("unsure", "unsure"),
        ("", "unsure"),
        ("banana", "unsure"),
        ("I cannot see the image", "unsure"),
        ("person or vehicle", "unsure"),
        ("\x00\xff", "unsure"),
        ("person " * 10_000, "unsure"),
    ],
)
def test_ac4_answer_parsing_and_unparseable_is_unsure(text: str, expected: str) -> None:
    for word, answer in judge.ANSWER_WORDS.items():
        assert judge.parse_answer(word) == answer
    assert judge.parse_answer(text) == expected


def _pairs(counts: dict[tuple[str, str], int]) -> list[tuple[Any, Any]]:
    return [pair for pair, n in counts.items() for _ in range(n)]


def test_ac4_scores_count_and_estimate_precision_on_three_mixes() -> None:
    assert judge.MIXES == (0.80, 0.90, 0.95)
    perfect = judge.score(
        _pairs(
            {
                ("person", "person"): 50,
                ("in_vehicle", "in_vehicle"): 10,
                ("not_person", "not_person"): 40,
            }
        )
    )
    assert perfect.n == 100 and perfect.accuracy == 1.0 and perfect.unsure_rate == 0.0
    assert all(abs(e) < 1e-9 for e in perfect.precision_error.values())
    assert set(perfect.precision_error) == set(judge.MIXES)
    yes_man = judge.score(
        _pairs(
            {("person", "person"): 50, ("in_vehicle", "person"): 10, ("not_person", "person"): 40}
        )
    )
    assert yes_man.accuracy == pytest.approx(50 / 100)
    for mix, error in yes_man.precision_error.items():
        assert error == pytest.approx(100 * (1 - mix))  # says 100% whatever the truth
    shy = judge.score(
        _pairs(
            {
                ("person", "unsure"): 10,
                ("person", "person"): 40,
                ("not_person", "not_person"): 45,
                ("not_person", "person"): 5,
            }
        )
    )
    assert shy.unsure_rate == pytest.approx(0.10)
    assert shy.accuracy == pytest.approx(85 / 90)
    assert shy.confusion["person"]["unsure"] == 10
    assert shy.confusion["not_person"]["person"] == 5


def test_ac4_bakeoff_cli_prints_counts_and_timings_only(
    capsys: pytest.CaptureFixture[str], offline: None
) -> None:
    counts: list[tuple[goldset.Label, int]] = [("person", 6), ("in_vehicle", 2), ("not_person", 4)]
    labels = [label for label, n in counts for _ in range(n)]
    answers: list[judge.Answer] = [label for label in labels]
    crops = [(label, np.full((64, 32, 3), 10 * i, np.uint8)) for i, label in enumerate(labels)]

    class Oracle:
        """Answers every crop correctly, in order."""

        def __init__(self) -> None:
            self.i = 0

        def classify(self, image: npt.NDArray[np.uint8]) -> judge.Answer:
            answer = answers[self.i]
            self.i += 1
            return answer

        def close(self) -> None:
            pass

    code = judge.main(
        ["--bakeoff"], open_judge=lambda candidate: Oracle(), crops=lambda: iter(crops)
    )
    out = capsys.readouterr().out
    assert code == 0
    for c in judge.CANDIDATES.values():
        assert c.name in out
    for needle in ("accuracy", "unsure", "seconds per crop", "80%", "90%", "95%"):
        assert needle in out, needle
    assert "http" not in out and ".jpg" not in out and "oi-" not in out


def test_ac4_module_runs_as_a_script() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "wearreport.tools.judge", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--bakeoff" in proc.stdout


# AC5: the quality bar -----------------------------------------------------------------


def _scores(accuracy: float, unsure: float, error: float) -> judge.Scores:
    return judge.Scores(
        n=400,
        confusion={},
        accuracy=accuracy,
        unsure_rate=unsure,
        precision_error=dict.fromkeys(judge.MIXES, error),
    )


def test_ac5_quality_bar() -> None:
    seconds = 3600 / 300  # exactly 60 minutes for 300 crops
    assert judge.passes(_scores(0.95, 0.10, 3.0), seconds)
    assert judge.passes(_scores(0.99, 0.0, -3.0), seconds)
    assert not judge.passes(_scores(0.949, 0.05, 0.0), 1.0)
    assert not judge.passes(_scores(0.99, 0.101, 0.0), 1.0)
    assert not judge.passes(_scores(0.99, 0.05, 3.01), 1.0)
    assert not judge.passes(_scores(0.99, 0.05, -3.01), 1.0)
    assert not judge.passes(_scores(0.99, 0.05, 0.0), seconds + 0.01)


def test_ac5_two_model_agreement() -> None:
    assert judge.agree("person", "person") == "person"
    assert judge.agree("not_person", "not_person") == "not_person"
    assert judge.agree("person", "not_person") == "unsure"
    assert judge.agree("person", "in_vehicle") == "unsure"
    assert judge.agree("unsure", "person") == "unsure"


def test_ac5_a_chosen_model_is_pinned() -> None:
    assert judge.CHOSEN in judge.CANDIDATES


def test_ac5_judge_workflow_is_manual_read_only_and_pinned() -> None:
    wf = JUDGE_WORKFLOW.read_text()
    on = wf.split("\non:", 1)[1].split("\n\n", 1)[0]
    assert "workflow_dispatch" in on
    for trigger in ("push", "pull_request", "schedule", "workflow_run", "issue"):
        assert trigger not in on, trigger
    assert "pull_request_target" not in wf
    assert re.search(r"^permissions:\n  contents: read\s*$", wf, flags=re.MULTILINE)
    assert "contents: write" not in wf and "secrets." not in wf
    assert "runs-on: ubuntu-24.04" in wf and "ubuntu-latest" not in wf
    assert "timeout-minutes:" in wf
    uses = re.findall(r"uses:\s*(\S+)", wf)
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u) for u in uses), uses
    assert re.search(rf'{REQUIRE_JUDGE}: "?1"?', wf)
    for needle in ("scripts/fetch_judge_model.sh", "scripts/fetch_goldset.sh", "--bakeoff"):
        assert needle in wf, needle


def test_ac5_ci_neither_downloads_nor_requires_the_weights() -> None:
    ci = CI_WORKFLOW.read_text()
    assert REQUIRE_JUDGE not in ci
    assert "fetch_judge_model" not in ci and "fetch_goldset" not in ci


# AC6: controls for live runs ----------------------------------------------------------


def _control_items(manifest: goldset.Manifest) -> list[goldset.Item]:
    return [item for item in manifest.items if item.control is not None]


def test_ac6_thirty_to_sixty_small_balanced_controls(manifest: goldset.Manifest) -> None:
    files = sorted(p for p in CONTROLS.iterdir() if not p.name.startswith("."))
    assert 30 <= len(files) <= 60
    items = _control_items(manifest)
    assert sorted(ROOT / str(item.control) for item in items) == files
    for item in items:
        path = ROOT / str(item.control)
        data = path.read_bytes()
        assert len(data) <= 20 * 1024, path.name
        assert data[:3] == b"\xff\xd8\xff", path.name
        assert hashlib.sha256(data).hexdigest() == item.control_sha256, path.name
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[0] <= 2 * 80 + 2, path.name
    counts = [sum(item.label == label for item in items) for label in goldset.LABELS]
    assert min(counts) >= 10 and max(counts) - min(counts) <= 2, counts


def test_ac6_controls_are_recorded_in_the_licence_table(manifest: goldset.Manifest) -> None:
    text = (ROOT / "fixtures" / "LICENSES.md").read_text()
    rows = {}
    for line in [ln for ln in text.splitlines() if ln.strip().startswith("|")][2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == 3
        rows[cells[2]] = (cells[0], cells[1])
    assert "fixtures/goldset/manifest.json" in rows
    assert ALLOWED_LICENCE.fullmatch(rows["fixtures/goldset/manifest.json"][1])
    for item in _control_items(manifest):
        source = manifest.sources[item.source]
        source_cell, licence = rows[str(item.control)]
        assert source.page in source_cell and source.author in source_cell, item.control
        assert licence == source.license, item.control


def test_ac6_controls_are_reproducible_from_their_sources(manifest: goldset.Manifest) -> None:
    if not _sources_present(manifest):
        _require("the gold-set source images")
    for item in _control_items(manifest):
        committed = cv2.imdecode(
            np.frombuffer((ROOT / str(item.control)).read_bytes(), np.uint8), cv2.IMREAD_COLOR
        )
        regenerated = goldset.degrade_item(manifest, item).frame
        assert committed is not None, item.id
        assert committed.shape == regenerated.shape, item.id
        diff = np.abs(committed.astype(np.int16) - regenerated.astype(np.int16))
        assert float(diff.mean()) <= 1.0, item.id


# AC7: licences -----------------------------------------------------------------------


def test_ac7_runner_dependency_declared_and_license_check_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0].lower() for d in deps}
    assert "llama-cpp-python" in names
    assert license_check.main([]) == 0
    out = capsys.readouterr().out
    assert "clean" in out and "llama_cpp_python" in out.replace("-", "_")
    lock = (ROOT / "uv.lock").read_text().lower()
    for forbidden in ("ultralytics", "huggingface-hub", "torch"):
        assert not re.search(rf'^name = "{forbidden}"$', lock, flags=re.MULTILINE), forbidden


# AC8: guards -------------------------------------------------------------------------


def test_ac8_privacy_guard_unchanged_scans_and_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert not privacy_guard.BINARY_WRITE_ALLOWLIST
    assert privacy_guard.IMAGE_WRITE_EXEMPTION == "engine/wearreport/tools/spotcheck.py"
    scanned = {p.relative_to(ROOT).as_posix() for p in privacy_guard.engine_files(ROOT)}
    assert set(NEW_MODULES) <= scanned
    assert privacy_guard.main(["--root", str(ROOT)]) == 0


def test_ac8_gold_images_are_read_as_bytes_and_decoded_in_memory() -> None:
    source = (ROOT / "engine" / "wearreport" / "tools" / "goldset.py").read_text()
    assert "imdecode" in source
    assert "imread" not in source and "VideoCapture" not in source
    for module in NEW_MODULES:
        text = (ROOT / module).read_text()
        assert privacy_guard.scan_source(text, module) == []
        assert "imwrite" not in text and "tofile" not in text


def test_ac8_gold_pipeline_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _synthetic_frame(600, 400)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    body = encoded.tobytes()
    source = goldset.Source(
        id="synthetic",
        url="https://example.org/synthetic",
        page="https://example.org/",
        author="test",
        license="CC0 1.0",
        sha256=hashlib.sha256(body).hexdigest(),
        width=400,
        height=600,
    )
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "synthetic").write_bytes(body)
    before = sorted(p for p in tmp_path.rglob("*"))
    monkeypatch.chdir(tmp_path)
    image = goldset.read_source(source, source_dir)
    degraded = goldset.degrade(image, (150, 80, 181, 201), 30, 40)
    rendered = goldset.render_crop(degraded.frame, degraded.box, 1)
    assert rendered.ndim == 3
    assert sorted(p for p in tmp_path.rglob("*")) == before
    (source_dir / "synthetic").write_bytes(body + b"x")
    with pytest.raises(goldset.GoldsetError):
        goldset.read_source(source, source_dir)  # a file that does not match its pin
