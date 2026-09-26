"""The spot-check judge: an open-weights vision-language model, run locally on the CPU,
that tells whether a detection crop shows a person, and the bake-off that chooses it.

  python -m wearreport.tools.judge --bakeoff [--models A,B] [--subset all|screen]
      [--limit N] [--threads 4]

`--bakeoff` runs every candidate in CANDIDATES (or those named) over the gold set
(`wearreport.tools.goldset`), each in a fresh process, and prints for each model: the
accuracy on its confident answers, its unsure rate, a per-class confusion table, the
error of the precision it would estimate on three realistic mixes (true precision 80%,
90% and 95%), the seconds per crop, and the peak memory. Then the same for every pair of
models under two-model agreement, and which model passes the quality bar. The output is
counts and timings only: no item, file or URL.

The judge (`Judge`) sees what a reviewer sees: the crop that the spot-check tool renders,
enlarged with its box and number drawn on it. It is asked PROMPT, a fixed question, and
answers with one word, parsed by `parse_answer` into person, in_vehicle, not_person or
unsure. Anything else counts as unsure.

Local and private (AGENTS.md INV-1 exception (a)). The runtime is llama.cpp through
llama-cpp-python, on the CPU only. Images are passed to it as pixel arrays in memory
(`mtmd_bitmap_init`); none of its loaders that take a path or a URL is used, and nothing is
written to disk. Neither llama.cpp nor llama-cpp-python has telemetry; the one network
feature of the Python package (`Llama.from_pretrained`) goes through huggingface_hub,
which is not installed. The Hugging Face and generic opt-out variables in TELEMETRY_ENV
are set before the runtime is imported, all the same, and the import is refused if the
runtime was loaded without them. llama.cpp's own environment switches (GGML_*, LLAMA_*,
MTMD_*: debug dumps, a backend library path) are removed before it loads.

No model is chosen (CHOSEN is None): none passes the quality bar. The weights of the
bake-off candidates come from `scripts/fetch_judge_model.sh --all` (or `--model NAME`).
A model is opened only if both of its files match their pinned SHA-256: each file is
opened once, hashed through that descriptor, and loaded through the same descriptor
(/proc/self/fd/N), so it cannot be swapped between the check and the load.
"""

from __future__ import annotations

import os
import sys

TELEMETRY_ENV = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "DO_NOT_TRACK": "1",
}

if "llama_cpp" in sys.modules and any(os.environ.get(k) != v for k, v in TELEMETRY_ENV.items()):
    raise ImportError(
        "llama_cpp was imported before wearreport.tools.judge, so its telemetry "
        "settings are not in effect"
    )
os.environ.update(TELEMETRY_ENV)
# llama.cpp reads switches from the environment when it loads and runs: GGML_BACKEND_PATH
# loads a backend library from any path, and the *_DEBUG and trace switches print model
# internals (image embeddings among them). None of them is ever wanted here.
RUNTIME_ENV_PREFIXES = ("GGML_", "LLAMA_", "MTMD_")
for _key in [k for k in os.environ if k.startswith(RUNTIME_ENV_PREFIXES)]:
    del os.environ[_key]

import argparse  # noqa: E402
import ctypes  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Iterable, Iterator, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from types import ModuleType  # noqa: E402
from typing import Any, BinaryIO, Literal, Protocol, get_args  # noqa: E402

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

from wearreport._cv import cv2  # noqa: E402
from wearreport.tools import goldset  # noqa: E402

Answer = Literal["person", "in_vehicle", "not_person", "unsure"]
ANSWERS: tuple[Answer, ...] = get_args(Answer)
Image = npt.NDArray[np.uint8]

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / ".models" / "judge"
DEFAULT_THREADS = 4

# The question, and the one-word answers it offers.
PROMPT = (
    "This picture is a crop from a low-resolution street camera. "
    "Look at what is inside the green box.\n"
    "Reply with one word:\n"
    "person - a person on foot or riding a bicycle\n"
    "vehicle - a person inside a car, bus, van or other vehicle\n"
    "other - anything else: a pole, bin, sign, bollard, shadow, or a picture, statue "
    "or model of a person\n"
    "unsure - you cannot tell"
)
ANSWER_WORDS: dict[str, Answer] = {
    "person": "person",
    "vehicle": "in_vehicle",
    "other": "not_person",
    "unsure": "unsure",
}
MAX_ANSWER_CHARS = 64
MAX_ANSWER_TOKENS = 6
MEDIA = "<__media__>"  # libmtmd's marker for where the image goes
TEMPLATES = {
    "chatml": "<|im_start|>user\n{media}{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "chatml-nothink": (
        "<|im_start|>user\n{media}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    ),
    "chatml-newline": "<|im_start|>user\n{media}\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "smolvlm": "<|im_start|>User:{media}{prompt}<end_of_utterance>\nAssistant:",
}
N_CTX = 4096
MAX_IMAGE_SIDE = 2048

# The quality bar.
MIXES = (0.80, 0.90, 0.95)  # true precision of the realistic mixes
MIX_IN_VEHICLE_SHARE = 0.15  # of the true positives in a mix, the share inside vehicles
MIN_ACCURACY = 0.95
MAX_UNSURE = 0.10
MAX_PRECISION_ERROR = 3.0  # points
BAR_CROPS = 300
BAR_SECONDS = 60 * 60
POSITIVE: frozenset[str] = frozenset({"person", "in_vehicle"})


class JudgeError(RuntimeError):
    """The model cannot be opened or failed on an image."""


# Candidates -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One pinned open-weights model: a language model and its vision projector (GGUF)."""

    name: str
    family: str
    params_b: float  # parameters, in billions, from the model card
    licence: str  # of the weights, from the model card
    card: str  # the original model card
    repo: str  # the Hugging Face repository of the GGUF files
    revision: str  # a commit of that repository
    model_file: str
    model_sha256: str
    model_bytes: int
    mmproj_file: str
    mmproj_sha256: str
    mmproj_bytes: int
    template: str  # key of TEMPLATES


CANDIDATES: dict[str, Candidate] = {
    c.name: c
    for c in (
        Candidate(
            name="qwen3.5-2b",
            family="Qwen3.5",
            params_b=2.27,
            licence="Apache-2.0",
            card="https://huggingface.co/Qwen/Qwen3.5-2B",
            repo="bartowski/Qwen_Qwen3.5-2B-GGUF",
            revision="7d26695454df6de5fbcce2e58681e62dae06ce43",
            model_file="Qwen_Qwen3.5-2B-Q8_0.gguf",
            model_sha256="be647507ce6cde229b838924d47bfff9763171105563f7f908670dae57c4dbe2",
            model_bytes=2080140384,
            mmproj_file="mmproj-Qwen_Qwen3.5-2B-f16.gguf",
            mmproj_sha256="044a0ea136cca70711ae16e23b24d754b44eab6f2462d187aee4d7c7a9503d36",
            mmproj_bytes=668227136,
            template="chatml-nothink",
        ),
        Candidate(
            name="qwen3.5-4b",
            family="Qwen3.5",
            params_b=4.66,
            licence="Apache-2.0",
            card="https://huggingface.co/Qwen/Qwen3.5-4B",
            repo="bartowski/Qwen_Qwen3.5-4B-GGUF",
            revision="4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
            model_file="Qwen_Qwen3.5-4B-Q8_0.gguf",
            model_sha256="5c74c0ede371924357dff0cb6ba145bd67208b9b2389ded681adfff3f7608db7",
            model_bytes=4622131168,
            mmproj_file="mmproj-Qwen_Qwen3.5-4B-f16.gguf",
            mmproj_sha256="659b59dd44b73b1cd34af6cc424669484b06dc80f4340adf8ea84ad776eef813",
            mmproj_bytes=672423488,
            template="chatml-nothink",
        ),
        Candidate(
            name="qwen3-vl-2b",
            family="Qwen3-VL",
            params_b=2.13,
            licence="Apache-2.0",
            card="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
            repo="Qwen/Qwen3-VL-2B-Instruct-GGUF",
            revision="52d6c8ffea26cc873ac5ad116f8631268d7eb503",
            model_file="Qwen3VL-2B-Instruct-Q8_0.gguf",
            model_sha256="1e8db19207c8ce0733ddd78c2eff8a9e22c27c82f4443df94c25792ed8fe04f2",
            model_bytes=1834427424,
            mmproj_file="mmproj-Qwen3VL-2B-Instruct-F16.gguf",
            mmproj_sha256="c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5",
            mmproj_bytes=819394848,
            template="chatml",
        ),
        Candidate(
            name="internvl3.5-2b",
            family="InternVL3.5",
            params_b=2.35,
            licence="Apache-2.0",
            card="https://huggingface.co/OpenGVLab/InternVL3_5-2B",
            repo="bartowski/OpenGVLab_InternVL3_5-2B-GGUF",
            revision="09023986543a68f5caaa389f64b0e0256fe22565",
            model_file="OpenGVLab_InternVL3_5-2B-Q8_0.gguf",
            model_sha256="6997c6e3a1fe5920ac1429a21a3ec15d545e14eb695ee3656834859e617800b5",
            model_bytes=2165036128,
            mmproj_file="mmproj-OpenGVLab_InternVL3_5-2B-f16.gguf",
            mmproj_sha256="e83ba6e675b747f7801557dc24594f43c17a7850b6129d4972d55e3e9b010359",
            mmproj_bytes=636106144,
            template="chatml-newline",
        ),
        Candidate(
            name="smolvlm2-2.2b",
            family="SmolVLM2",
            params_b=2.25,
            licence="Apache-2.0",
            card="https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            repo="ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
            revision="1bc3c9f74ceafd4c8d4411cc9cf188bba3798f91",
            model_file="SmolVLM2-2.2B-Instruct-Q8_0.gguf",
            model_sha256="c850ffa51b0708be8911766e1d35e8e71365e987c8efb2513a7f237baade074f",
            model_bytes=1927933984,
            mmproj_file="mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
            mmproj_sha256="db9a3a1648cab1ebc3af4a2b0c8145dd8faebf6f7dd7b16e7dc1842229f14ac4",
            mmproj_bytes=872303680,
            template="smolvlm",
        ),
    )
}
# No model is chosen: no candidate and no two-model agreement passes the quality bar (T-029
# bake-off: the best, qwen3.5-4b, reaches 76.2% accuracy on confident answers against the
# 95% bar). Until one does, nothing names a judge: `--models chosen` is refused and
# scripts/fetch_judge_model.sh without --model or --all exits non-zero.
CHOSEN: str | None = None
NO_CHOSEN_MODEL = "no judge model is chosen: no candidate passes the quality bar"


def download_url(candidate: Candidate, name: str) -> str:
    """Where scripts/fetch_judge_model.sh downloads one of a candidate's files."""
    return f"https://huggingface.co/{candidate.repo}/resolve/{candidate.revision}/{name}"


# Answers ----------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z_]+")


def parse_answer(text: str) -> Answer:
    """The judge's answer: its first word must be one of ANSWER_WORDS, and no other answer
    word may follow. Anything else, including an empty or very long reply, is unsure."""
    if not isinstance(text, str) or len(text) > MAX_ANSWER_CHARS:
        return "unsure"
    words = _WORD.findall(text.lower())
    if not words or words[0] not in ANSWER_WORDS:
        return "unsure"
    named = {ANSWER_WORDS[w] for w in words if w in ANSWER_WORDS}
    return ANSWER_WORDS[words[0]] if len(named) == 1 else "unsure"


def agree(a: Answer, b: Answer) -> Answer:
    """Two-model agreement: the shared answer, or unsure when they differ."""
    return a if a == b else "unsure"


# The runtime ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Runtime:
    llama_cpp: ModuleType
    mtmd: ModuleType


_RUNTIME: Runtime | None = None


def _drop_log(level: int, text: bytes | None, user_data: int | None) -> None:
    """llama.cpp's log callback: drop everything (it logs sizes and timings, never pixels)."""


_LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_quiet = _LOG_CALLBACK(_drop_log)  # kept referenced for as long as the library may call it


def load_runtime() -> Runtime:
    """Import llama-cpp-python (the settings above are already in the environment), start
    its CPU backend and silence its logs. Idempotent."""
    global _RUNTIME
    if _RUNTIME is None:
        import llama_cpp
        from llama_cpp import mtmd_cpp

        llama_cpp.llama_log_set(_quiet, ctypes.c_void_p(0))
        mtmd_cpp.mtmd_log_set(_quiet, ctypes.c_void_p(0))
        mtmd_cpp.mtmd_helper_log_set(_quiet, ctypes.c_void_p(0))
        llama_cpp.llama_backend_init()
        _RUNTIME = Runtime(llama_cpp=llama_cpp, mtmd=mtmd_cpp)
    return _RUNTIME


class Classifier(Protocol):
    def classify(self, image: Image) -> Answer: ...

    def close(self) -> None: ...


def _open_verified(path: Path, sha256: str, size: int) -> BinaryIO:
    """`path` opened for reading, once its content has matched `sha256`."""
    try:
        fh = open(path, "rb")  # noqa: SIM115  (the caller closes it)
    except OSError as exc:
        raise JudgeError(
            f"cannot open {path.name}: {exc.strerror}; run scripts/fetch_judge_model.sh"
        ) from None
    try:
        digest = hashlib.sha256()
        total = 0
        while chunk := fh.read(1 << 22):
            digest.update(chunk)
            total += len(chunk)
            if total > size:
                break
        if total != size or digest.hexdigest() != sha256:
            raise JudgeError(f"{path.name} does not match its pinned SHA-256")
        fh.seek(0)
    except BaseException:
        fh.close()
        raise
    return fh


def validate_image(image: object) -> Image:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("an image must be a uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"an image must be HxWx3 (BGR), not shape {image.shape}")
    height, width = image.shape[:2]
    if not (0 < height <= MAX_IMAGE_SIDE and 0 < width <= MAX_IMAGE_SIDE):
        raise ValueError(f"an image must be 1 to {MAX_IMAGE_SIDE} pixels on each side")
    return image


class Judge:
    """One candidate model, loaded on the CPU, answering PROMPT about BGR images."""

    def __init__(
        self,
        candidate: Candidate,
        *,
        model_dir: Path = MODEL_DIR,
        threads: int = DEFAULT_THREADS,
    ) -> None:
        if not 1 <= threads <= 64:
            raise ValueError("threads must be from 1 to 64")
        self.candidate = candidate
        self._template = TEMPLATES[candidate.template]
        self._llm: Any = None
        self._ctx: Any = None
        files = [
            _open_verified(
                model_dir / candidate.model_file, candidate.model_sha256, candidate.model_bytes
            ),
        ]
        try:
            files.append(
                _open_verified(
                    model_dir / candidate.mmproj_file,
                    candidate.mmproj_sha256,
                    candidate.mmproj_bytes,
                )
            )
            self._load(files[0].fileno(), files[1].fileno(), threads)
        except BaseException:
            self.close()
            raise
        finally:
            for fh in files:
                fh.close()

    def _load(self, model_fd: int, mmproj_fd: int, threads: int) -> None:
        rt = load_runtime()
        try:
            self._llm = rt.llama_cpp.Llama(
                model_path=f"/proc/self/fd/{model_fd}",
                n_gpu_layers=0,
                n_ctx=N_CTX,
                n_batch=512,
                n_threads=threads,
                n_threads_batch=threads,
                seed=0,
                verbose=False,
            )
            params = rt.mtmd.mtmd_context_params_default()
            params.use_gpu = False
            params.n_threads = threads
            params.print_timings = False
            self._ctx = rt.mtmd.mtmd_init_from_file(
                f"/proc/self/fd/{mmproj_fd}".encode(), self._llm.model, params
            )
        except (ValueError, OSError, RuntimeError) as exc:
            raise JudgeError(f"llama.cpp cannot load {self.candidate.name}: {exc}") from None
        if not self._ctx or not rt.mtmd.mtmd_support_vision(self._ctx):
            raise JudgeError(f"{self.candidate.mmproj_file} is not a vision projector")

    def classify(self, image: Image) -> Answer:
        """The model's answer about the box drawn in `image` (HxWx3 uint8, BGR)."""
        return parse_answer(self.reply(image))

    def reply(self, image: Image) -> str:
        """The model's raw reply (at most MAX_ANSWER_TOKENS tokens, greedy)."""
        if self._llm is None or self._ctx is None:
            raise JudgeError("the judge is closed")
        pixels = validate_image(image)
        rgb = np.ascontiguousarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        rt = load_runtime()
        m, llm = rt.mtmd, self._llm
        text = self._template.format(media=MEDIA, prompt=PROMPT).encode()
        height, width = rgb.shape[:2]
        bitmap = m.mtmd_bitmap_init(
            width, height, rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        )
        chunks = m.mtmd_input_chunks_init()
        try:
            if not bitmap or not chunks:
                raise JudgeError("llama.cpp cannot take the image")
            prompt = m.mtmd_input_text()
            prompt.text = text
            prompt.text_len = len(text)
            prompt.add_special = True
            prompt.parse_special = True
            bitmaps = (m.mtmd_bitmap_p_ctypes * 1)(bitmap)
            if m.mtmd_tokenize(self._ctx, chunks, ctypes.byref(prompt), bitmaps, 1) != 0:
                raise JudgeError("llama.cpp cannot tokenize the prompt")
            llm.reset()
            llm._ctx.kv_cache_clear()
            n_past = rt.llama_cpp.llama_pos(0)
            status = m.mtmd_helper_eval_chunks(
                self._ctx,
                llm._ctx.ctx,
                chunks,
                rt.llama_cpp.llama_pos(0),
                rt.llama_cpp.llama_seq_id(0),
                llm.n_batch,
                True,
                ctypes.byref(n_past),
            )
            if status != 0:
                raise JudgeError(f"llama.cpp failed on the image (status {status})")
            llm.n_tokens = n_past.value
            return self._greedy(rt)
        finally:
            if chunks:
                m.mtmd_input_chunks_free(chunks)
            if bitmap:
                m.mtmd_bitmap_free(bitmap)

    def _greedy(self, rt: Runtime) -> str:
        llm = self._llm
        vocab = llm._model.vocab
        tokens: list[int] = []
        for _ in range(MAX_ANSWER_TOKENS):
            logits = rt.llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1)
            token = int(np.ctypeslib.as_array(logits, shape=(llm.n_vocab(),)).argmax())
            if rt.llama_cpp.llama_vocab_is_eog(vocab, token):
                break
            tokens.append(token)
            llm.eval([token])
        return bytes(llm.detokenize(tokens)).decode("utf-8", errors="replace")

    def close(self) -> None:
        if self._ctx:
            load_runtime().mtmd.mtmd_free(self._ctx)
        self._ctx = None
        if self._llm is not None:
            self._llm.close()
        self._llm = None


# Scores -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scores:
    n: int
    confusion: dict[str, dict[str, int]]  # truth -> answer -> count
    accuracy: float  # on confident (not unsure) answers
    unsure_rate: float
    precision_error: dict[float, float]  # true precision -> estimate minus truth, in points


def _estimate_error(confusion: dict[str, dict[str, int]], mix: float) -> float:
    """Estimated minus true precision (points) on a mix with true precision `mix`.

    Precision is the share of boxes that show a person, on foot, cycling or in a vehicle
    (the spot-check's precision_person). The judge's answers on each class are taken at
    their gold-set rates: positives are MIX_IN_VEHICLE_SHARE in vehicles, negatives are
    the gold set's not_person items. The estimate counts confident answers only.
    """
    weights = {
        "person": mix * (1 - MIX_IN_VEHICLE_SHARE),
        "in_vehicle": mix * MIX_IN_VEHICLE_SHARE,
        "not_person": 1 - mix,
    }
    totals = {c: sum(confusion.get(c, {}).values()) for c in goldset.LABELS}
    positives = [c for c in ("person", "in_vehicle") if totals[c]]
    if not positives or not totals["not_person"]:
        return math.nan
    share = sum(weights[c] for c in positives)
    for c in positives:  # a class with no gold items passes its weight to the other
        weights[c] = mix * weights[c] / share
    said_person = confident = 0.0
    for c in goldset.LABELS:
        if not totals[c]:
            continue
        row = confusion[c]
        said_person += weights[c] * sum(row.get(a, 0) for a in POSITIVE) / totals[c]
        confident += weights[c] * (totals[c] - row.get("unsure", 0)) / totals[c]
    if confident == 0:
        return math.nan
    return 100 * (said_person / confident - mix)


def score(pairs: Iterable[tuple[goldset.Label, Answer]]) -> Scores:
    """Scores for (truth, answer) pairs."""
    confusion: dict[str, dict[str, int]] = {c: dict.fromkeys(ANSWERS, 0) for c in goldset.LABELS}
    for truth, answer in pairs:
        confusion[truth][answer] += 1
    n = sum(sum(row.values()) for row in confusion.values())
    unsure = sum(row["unsure"] for row in confusion.values())
    correct = sum(confusion[c][c] for c in goldset.LABELS)
    confident = n - unsure
    return Scores(
        n=n,
        confusion=confusion,
        accuracy=correct / confident if confident else 0.0,
        unsure_rate=unsure / n if n else 1.0,
        precision_error={mix: _estimate_error(confusion, mix) for mix in MIXES},
    )


def passes(scores: Scores, seconds_per_crop: float) -> bool:
    """The quality bar: accurate, rarely unsure, an unbiased precision estimate on every
    mix, and 300 crops within an hour."""
    return (
        scores.accuracy >= MIN_ACCURACY
        and scores.unsure_rate <= MAX_UNSURE
        and all(abs(e) <= MAX_PRECISION_ERROR for e in scores.precision_error.values())
        and BAR_CROPS * seconds_per_crop <= BAR_SECONDS
    )


def _failures(scores: Scores, seconds_per_crop: float) -> list[str]:
    out = []
    if scores.accuracy < MIN_ACCURACY:
        out.append(f"accuracy below {MIN_ACCURACY:.0%}")
    if scores.unsure_rate > MAX_UNSURE:
        out.append(f"unsure above {MAX_UNSURE:.0%}")
    if not all(abs(e) <= MAX_PRECISION_ERROR for e in scores.precision_error.values()):
        out.append(f"precision error above {MAX_PRECISION_ERROR:g} points")
    if BAR_CROPS * seconds_per_crop > BAR_SECONDS:
        out.append(f"{BAR_CROPS} crops take over {BAR_SECONDS // 60} minutes")
    return out


# The bake-off -----------------------------------------------------------------------------

# (truth, image), or (truth, image, the item's height_px) from the gold set.
Crop = tuple[goldset.Label, Image] | tuple[goldset.Label, Image, int]
Crops = Callable[[], Iterable[Crop]]
HEIGHT_BANDS = ((15, 30), (31, 55), (56, 80))  # person height, pixels


@dataclass(slots=True)
class Run:
    name: str
    answers: list[tuple[goldset.Label, Answer]] = field(default_factory=list)
    seconds: float = 0.0  # classifying, in total
    load_seconds: float = 0.0
    peak_rss_mib: float = 0.0
    error: str | None = None
    heights: list[int] = field(default_factory=list)  # height_px per answer, 0 if unknown


def evaluate(name: str, open_judge: Callable[[], Classifier], crops: Crops) -> Run:
    """Open a judge, classify every crop, and time it."""
    run = Run(name)
    start = time.perf_counter()
    try:
        judge = open_judge()
    except (JudgeError, goldset.GoldsetError, ValueError) as exc:
        run.error = str(exc)
        return run
    run.load_seconds = time.perf_counter() - start
    try:
        for crop in crops():
            truth, image = crop[0], crop[1]
            t0 = time.perf_counter()
            answer = judge.classify(image)
            run.seconds += time.perf_counter() - t0
            run.answers.append((truth, answer))
            run.heights.append(crop[2] if len(crop) == 3 else 0)
    except (JudgeError, goldset.GoldsetError, ValueError) as exc:
        run.error = str(exc)
    finally:
        judge.close()
    run.peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return run


def _gold_crops(subset: str, limit: int | None) -> Crops:
    def crops() -> Iterator[Crop]:
        manifest = goldset.load_manifest()
        items = goldset.screening_subset(manifest) if subset == "screen" else manifest.items
        for item, image in goldset.iter_gold(manifest, items=items[:limit]):
            yield item.label, image, item.height_px

    return crops


def _child(name: str, subset: str, limit: int | None, threads: int) -> Run:
    """One candidate over the gold set, in a fresh process (so its memory is its own)."""
    candidate = CANDIDATES[name]
    return evaluate(name, lambda: Judge(candidate, threads=threads), _gold_crops(subset, limit))


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{100 * x:.1f}%"


def report(title: str, scores: Scores, run_seconds: float, extra: str = "") -> list[str]:
    per_crop = run_seconds / scores.n if scores.n else math.inf
    confident = scores.n - sum(r["unsure"] for r in scores.confusion.values())
    correct = sum(scores.confusion[c][c] for c in goldset.LABELS)
    unsure = scores.n - confident
    lines = [
        f"== {title}",
        f"crops {scores.n}, seconds per crop {per_crop:.2f}, "
        f"{BAR_CROPS} crops {BAR_CROPS * per_crop / 60:.1f} min{extra}",
        f"accuracy on confident answers {_pct(scores.accuracy)} ({correct}/{confident}), "
        f"unsure {_pct(scores.unsure_rate)} ({unsure}/{scores.n})",
        "confusion (rows: truth; columns: answer)",
        "              " + "".join(f"{a:>12}" for a in ANSWERS),
    ]
    for truth in goldset.LABELS:
        row = scores.confusion[truth]
        lines.append(f"  {truth:<12}" + "".join(f"{row[a]:>12}" for a in ANSWERS))
    errors = ", ".join(
        f"{mix:.0%}: " + ("n/a" if math.isnan(e) else f"{e:+.1f}")
        for mix, e in scores.precision_error.items()
    )
    lines.append(f"precision error (points) at true precision {errors}")
    failures = _failures(scores, per_crop)
    lines.append(
        "quality bar: " + ("PASS" if not failures else "FAIL (" + "; ".join(failures) + ")")
    )
    return lines


def _by_height(run: Run) -> list[str]:
    """Accuracy on confident answers per band of person height, when heights are known."""
    if not run.heights or 0 in run.heights:
        return []
    parts = []
    for low, high in HEIGHT_BANDS:
        answers = [
            (t, a) for (t, a), h in zip(run.answers, run.heights, strict=True) if low <= h <= high
        ]
        confident = [(t, a) for t, a in answers if a != "unsure"]
        correct = sum(t == a for t, a in confident)
        share = f"{100 * correct / len(confident):.1f}%" if confident else "n/a"
        parts.append(f"{low}-{high} px {share} ({correct}/{len(confident)})")
    return ["accuracy on confident answers by person height: " + ", ".join(parts)]


def bakeoff(
    names: Sequence[str],
    run_one: Callable[[str], Run],
    emit: Callable[[str], None] = print,
) -> None:
    """Run each candidate, report it as soon as it is done, then the pairs, then the
    verdict."""
    runs: list[Run] = []
    passing: list[tuple[float, str]] = []
    for name in names:
        run = run_one(name)
        runs.append(run)
        c = CANDIDATES[run.name]
        title = f"{c.name} ({c.family}, {c.params_b:g}B parameters, {c.licence})"
        if run.error is not None:
            for line in (f"== {title}", f"not run: {run.error}"):
                emit(line)
            continue
        scores = score(run.answers)
        extra = f", load {run.load_seconds:.1f} s, peak RAM {run.peak_rss_mib / 1024:.2f} GiB"
        for line in report(title, scores, run.seconds, extra) + _by_height(run):
            emit(line)
        if scores.n and passes(scores, run.seconds / scores.n):
            passing.append((scores.accuracy, run.name))
    complete = [r for r in runs if r.error is None and r.answers]
    for a, b in itertools.combinations(complete, 2):
        if [t for t, _ in a.answers] != [t for t, _ in b.answers]:
            continue
        pairs = [(t, agree(x, y)) for (t, x), (_, y) in zip(a.answers, b.answers, strict=True)]
        for line in report(
            f"agreement of {a.name} and {b.name}", score(pairs), a.seconds + b.seconds
        ):
            emit(line)
    if passing:
        emit(f"most accurate model that passes: {max(passing)[1]}")
    else:
        emit("no single model passes the quality bar")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m wearreport.tools.judge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bakeoff", action="store_true", help="run the candidates on the gold set")
    ap.add_argument(
        "--models",
        default=",".join(CANDIDATES),
        help="comma-separated names; chosen = CHOSEN (refused: none is chosen)",
    )
    ap.add_argument("--subset", choices=("all", "screen"), default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N crops only")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    return ap


def main(
    argv: Sequence[str] | None = None,
    *,
    open_judge: Callable[[Candidate], Classifier] | None = None,
    crops: Crops | None = None,
) -> int:
    """Run the bake-off. `open_judge` and `crops` replace the real models and gold set
    (tests); they run in this process."""
    args = build_parser().parse_args(argv)
    if not args.bakeoff:
        build_parser().print_help()
        return 2
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    if "chosen" in names and CHOSEN is None:
        print(f"judge: {NO_CHOSEN_MODEL}", file=sys.stderr)
        return 2
    names = [CHOSEN if n == "chosen" and CHOSEN is not None else n for n in names]
    unknown = [n for n in names if n not in CANDIDATES]
    if unknown or not names:
        print(f"judge: unknown candidate {unknown[0] if unknown else ''!r}", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("judge: --limit must be at least 1", file=sys.stderr)
        return 2

    run_one: Callable[[str], Run]
    if open_judge is not None or crops is not None:
        opener = open_judge or (lambda c: Judge(c, threads=args.threads))
        source = crops or _gold_crops(args.subset, args.limit)

        def run_one(name: str) -> Run:
            return evaluate(name, lambda: opener(CANDIDATES[name]), source)
    else:
        context = multiprocessing.get_context("spawn")

        def run_one(name: str) -> Run:
            with context.Pool(1) as pool:
                return pool.apply(_child, (name, args.subset, args.limit, args.threads))

    bakeoff(names, run_one, lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
