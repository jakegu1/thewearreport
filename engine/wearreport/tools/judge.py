"""The spot-check judge: a vision-language model that tells whether a detection crop shows
a person, and the bake-off that chooses it. The model is either open weights run locally
on the CPU (CANDIDATES) or a model hosted by Amazon Bedrock (HOSTED).

  python -m wearreport.tools.judge --bakeoff [--models A,B] [--subset all|screen]
      [--limit N] [--threads 4]
  python -m wearreport.tools.judge --bakeoff --backend bedrock --max-requests N
      [--models A,B] [--subset all|screen] [--limit N]

`--bakeoff` runs every candidate of the backend (or those named) over the gold set
(`wearreport.tools.goldset`), and prints for each model: the accuracy on its confident
answers, its unsure rate, a per-class confusion table, the error of the precision it would
estimate on three realistic mixes (true precision 80%, 90% and 95%), the seconds per crop,
and the peak memory (local) or the region, requests, tokens and measured cost (hosted).
The same follows for the held-out items, the gold set outside the fixed screening subset
on which the prompt may be tuned: a model passes only if the whole gold set and the
held-out items both pass the quality bar. Then every pair of models under two-model
agreement, and the verdict: the most accurate local model, or the cheapest hosted model
by measured cost per 300 crops, that passes. The output is counts, timings and costs
only: no item, file, URL, image or request.

Hosted models (`BedrockClassifier`) are called through the Bedrock Converse API with the
standard library only, one crop per request, as a PNG built in memory. The bearer token
is read from AWS_BEARER_TOKEN_BEDROCK and never printed, logged or put in an error. A
remote run refuses to start without `--max-requests N` and stops at N requests (retries
count). The backend sends only gold-set and control crops (`mark_licensed`): anything
else is refused before a request is made (AGENTS.md INV-1). Each request has a timeout;
throttling and server errors are retried at most MAX_RETRIES times with backoff, other
errors not at all; redirects are never followed.

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
import base64  # noqa: E402
import ctypes  # noqa: E402
import hashlib  # noqa: E402
import http.client  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import ssl  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402
import weakref  # noqa: E402
from collections.abc import Callable, Iterable, Iterator, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from types import ModuleType  # noqa: E402
from typing import Any, BinaryIO, Literal, Protocol, get_args, runtime_checkable  # noqa: E402

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


@dataclass(frozen=True, slots=True)
class HostedCandidate:
    """One vision model served by Amazon Bedrock, called with the Converse API. The model
    ID, the regions and the terms are from the model's page in the Bedrock documentation
    (`card`); the prices, in US dollars per million tokens on demand (Standard tier) in
    `region`, from the Amazon Bedrock pricing page. A model ID that starts with eu. or us.
    is a cross-region inference profile: Bedrock serves it within that geography."""

    name: str
    family: str
    model_id: str
    region: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    licence: str  # of the weights, or the terms under which Bedrock serves the model
    card: str
    backup: bool = False  # run only when no other hosted candidate passes


_BEDROCK_DOCS = "https://docs.aws.amazon.com/bedrock/latest/userguide/"
HOSTED: dict[str, HostedCandidate] = {
    c.name: c
    for c in (
        HostedCandidate(
            name="nova-2-lite",
            family="Amazon Nova 2",
            model_id="eu.amazon.nova-2-lite-v1:0",  # no in-region use; EU profile only
            region="eu-central-1",
            input_usd_per_mtok=0.429,
            output_usd_per_mtok=3.597,
            licence="AWS Service Terms (first-party model)",
            card=_BEDROCK_DOCS + "model-card-amazon-nova-2-lite.html",
        ),
        HostedCandidate(
            name="llama4-maverick",
            family="Meta Llama 4",
            model_id="us.meta.llama4-maverick-17b-instruct-v1:0",  # US profile only
            region="us-east-1",
            input_usd_per_mtok=0.24,
            output_usd_per_mtok=0.97,
            licence="Llama 4 Community License",
            card=_BEDROCK_DOCS + "model-card-meta-llama-4-maverick-17b-instruct.html",
        ),
        HostedCandidate(
            name="qwen3-vl-235b",
            family="Qwen3-VL",
            model_id="qwen.qwen3-vl-235b-a22b",
            region="eu-west-1",  # not offered in eu-central-1
            input_usd_per_mtok=0.62,
            output_usd_per_mtok=3.13,
            licence="Apache-2.0",
            card=_BEDROCK_DOCS + "model-card-qwen-qwen3-vl-235b-a22b.html",
        ),
        HostedCandidate(
            name="kimi-k2.5",
            family="Moonshot Kimi",
            model_id="moonshotai.kimi-k2.5",  # accepts image input
            region="eu-north-1",  # not offered in eu-central-1
            input_usd_per_mtok=0.72,
            output_usd_per_mtok=3.60,
            licence="Modified MIT License",
            card=_BEDROCK_DOCS + "model-card-moonshot-ai-kimi-k2-5.html",
        ),
        HostedCandidate(
            name="pixtral-large",
            family="Mistral Pixtral",
            model_id="eu.mistral.pixtral-large-2502-v1:0",  # no in-region use; EU profile
            region="eu-central-1",
            input_usd_per_mtok=2.00,
            output_usd_per_mtok=6.00,
            licence="Mistral AI terms for Amazon Bedrock (third-party model)",
            card=_BEDROCK_DOCS + "model-card-mistral-ai-pixtral-large.html",
            backup=True,
        ),
    )
}


def cost_usd(candidate: HostedCandidate, input_tokens: int, output_tokens: int) -> float:
    """What `input_tokens` and `output_tokens` cost at the candidate's published prices."""
    return (
        input_tokens * candidate.input_usd_per_mtok + output_tokens * candidate.output_usd_per_mtok
    ) / 1e6


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


@runtime_checkable
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


# Crops that may be sent to a hosted model -------------------------------------------------

# Only crops of the licensed gold set and the committed controls leave this machine. The
# code that renders them marks each one; the hosted backend refuses anything unmarked, a
# copy, or a marked array whose pixels have changed since.
_LICENSED: dict[int, tuple[weakref.ref[Image], str]] = {}


def _pixel_digest(image: Image) -> str:
    digest = hashlib.sha256(repr(image.shape).encode())
    digest.update(np.ascontiguousarray(image).data)
    return digest.hexdigest()


def mark_licensed(image: Image) -> Image:
    """A read-only copy of `image`, marked as a gold-set or control crop. Call it only on
    crops rendered from the licensed gold set or its committed controls."""
    frozen = np.array(validate_image(image), dtype=np.uint8, copy=True)
    frozen.flags.writeable = False
    key = id(frozen)
    _LICENSED[key] = (
        weakref.ref(frozen, lambda _ref, key=key: _LICENSED.pop(key, None)),  # type: ignore[misc]
        _pixel_digest(frozen),
    )
    return frozen


def is_licensed(image: object) -> bool:
    if not isinstance(image, np.ndarray):
        return False
    entry = _LICENSED.get(id(image))
    return entry is not None and entry[0]() is image and entry[1] == _pixel_digest(image)


# The hosted backend (Amazon Bedrock) ------------------------------------------------------

TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"  # noqa: S105  (the variable name, not a token)
REQUEST_TIMEOUT = 60.0  # seconds, for each connection, read and write
MAX_RETRIES = 3  # after the first attempt, on throttling and server errors only
BACKOFF_SECONDS = 2.0  # doubled after each retry
REMOTE_MAX_TOKENS = 10
PNG_SUFFIX = ".png"  # the format each crop is sent in, encoded in memory
MAX_RESPONSE_BYTES = 1 << 20
MAX_ERROR_CHARS = 200
FILTERED_STOPS = frozenset({"content_filtered", "guardrail_intervened"})
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})  # a local fake server (tests)
_MODEL_ID = re.compile(r"[A-Za-z0-9][\w.:-]{0,127}")
_TOKEN = re.compile(r"[\x21-\x7e]{1,8192}")


class RequestLimitReached(JudgeError):
    """The run has made its --max-requests requests."""


@dataclass(slots=True)
class Usage:
    """What a hosted model was asked and answered, from the API's usage field."""

    requests: int = 0  # every attempt, retries included
    input_tokens: int = 0
    output_tokens: int = 0
    missing: int = 0  # successful replies without a usable usage field


class RequestBudget:
    """At most `limit` requests for a whole run, shared by every hosted model in it."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("the request limit must be at least 1")
        self.limit = limit
        self.used = 0

    def take(self) -> None:
        if self.used >= self.limit:
            raise RequestLimitReached(f"request limit of {self.limit} reached")
        self.used += 1


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would carry the token to another address."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _endpoint(base: str, model_id: str) -> tuple[str, bool]:
    """The Converse URL for `model_id` at `base`, and whether `base` is on this machine."""
    parts = urllib.parse.urlsplit(base)
    local = parts.hostname in LOOPBACK
    if parts.scheme != "https" and not (parts.scheme == "http" and local):
        raise JudgeError("the Bedrock endpoint must be an https URL")
    if parts.path.strip("/") or parts.query or parts.fragment or parts.username:
        raise JudgeError("the Bedrock endpoint must be a bare origin")
    if not _MODEL_ID.fullmatch(model_id):
        raise JudgeError("malformed model ID")
    return f"{parts.scheme}://{parts.netloc}/model/{model_id}/converse", local


def _read_capped(response: Any) -> bytes:
    body = response.read(MAX_RESPONSE_BYTES + 1)
    if not isinstance(body, bytes) or len(body) > MAX_RESPONSE_BYTES:
        raise JudgeError(f"Bedrock sent a reply larger than {MAX_RESPONSE_BYTES} bytes")
    return body


def _reject_constant(name: str) -> object:
    raise ValueError(f"{name} is not a number")


def _load_json(raw: bytes) -> object:
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError, OverflowError) as exc:
        raise JudgeError(f"Bedrock sent a malformed reply ({type(exc).__name__})") from None


def _dict(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise JudgeError(f"Bedrock sent a malformed reply ({what} is not an object)")
    return value


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 10**9 else None


class BedrockClassifier:
    """A hosted candidate answering PROMPT about gold-set and control crops, one crop per
    Converse request, at temperature 0. `endpoint` replaces the Bedrock origin of the
    candidate's region (tests: a fake server on the loopback interface); `sleep` waits
    between retries."""

    def __init__(
        self,
        candidate: HostedCandidate,
        *,
        budget: RequestBudget,
        endpoint: str | None = None,
        timeout: float = REQUEST_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 < timeout <= 600:
            raise ValueError("timeout must be from 0 to 600 seconds")
        self.candidate = candidate
        self.usage = Usage()
        self._budget = budget
        self._timeout = timeout
        self._sleep = sleep
        base = endpoint or f"https://bedrock-runtime.{candidate.region}.amazonaws.com"
        self._url, local = _endpoint(base, candidate.model_id)
        token = os.environ.get(TOKEN_ENV, "")
        if not token:
            raise JudgeError(f"{TOKEN_ENV} is not set")
        if not _TOKEN.fullmatch(token):
            raise JudgeError(f"{TOKEN_ENV} is not a well-formed token")
        self._token: str | None = token
        # The system's proxy settings apply, except to a server on this machine.
        proxies = urllib.request.ProxyHandler({} if local else None)
        self._opener = urllib.request.build_opener(
            proxies,
            _NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        )

    def classify(self, image: Image) -> Answer:
        """The model's answer about the box drawn in `image`, a marked gold-set or control
        crop (HxWx3 uint8, BGR)."""
        if self._token is None:
            raise JudgeError("the judge is closed")
        pixels = validate_image(image)
        if not is_licensed(pixels):
            raise JudgeError("refusing an image that is not a gold-set or control crop")
        return self._answer(self._converse(self._body(pixels)))

    def close(self) -> None:
        self._token = None

    @staticmethod
    def _body(pixels: Image) -> bytes:
        ok, encoded = cv2.imencode(PNG_SUFFIX, pixels)  # in memory
        if not ok:
            raise JudgeError("cannot encode the crop")
        image = base64.b64encode(encoded.tobytes()).decode("ascii")
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "png", "source": {"bytes": image}}},
                        {"text": PROMPT},
                    ],
                }
            ],
            "inferenceConfig": {"maxTokens": REMOTE_MAX_TOKENS, "temperature": 0},
        }
        return json.dumps(request).encode()

    def _converse(self, body: bytes) -> bytes:
        for attempt in range(MAX_RETRIES + 1):
            status, raw, kind = self._send(body)
            if 200 <= status < 300:
                return raw
            retry = status == 429 or status >= 500 or kind == "ThrottlingException"
            if not retry or attempt == MAX_RETRIES:
                raise JudgeError(self._describe(status, kind, raw))
            self._sleep(BACKOFF_SECONDS * 2**attempt)
        raise AssertionError("unreachable")

    def _send(self, body: bytes) -> tuple[int, bytes, str]:
        """One request: the HTTP status, the body and the AWS error type."""
        if self._token is None:
            raise JudgeError("the judge is closed")
        self._budget.take()
        self.usage.requests += 1
        request = urllib.request.Request(  # noqa: S310  (the scheme is checked above)
            self._url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                return response.status, _read_capped(response), ""
        except urllib.error.HTTPError as exc:
            kind = (exc.headers.get("x-amzn-ErrorType") or "") if exc.headers else ""
            try:
                raw = _read_capped(exc)
            except (JudgeError, OSError, http.client.HTTPException):
                raw = b""
            finally:
                exc.close()
            return exc.code, raw, re.sub(r"[^A-Za-z]", "", kind.split(":", 1)[0])[:64]
        except TimeoutError:
            raise JudgeError(f"the request timed out after {self._timeout:g} s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise JudgeError(f"the request timed out after {self._timeout:g} s") from None
            raise JudgeError(f"cannot reach Bedrock ({type(exc.reason).__name__})") from None
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise JudgeError(f"the request failed ({type(exc).__name__})") from None

    def _describe(self, status: int, kind: str, raw: bytes) -> str:
        """An error for a failed request, with the service's message, never the token."""
        message = ""
        try:
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, dict):
                found = data.get("message") or data.get("Message")
                message = found if isinstance(found, str) else ""
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError, OverflowError):
            message = ""
        message = self._redact(message)
        text = f"Bedrock answered HTTP {status}" + (f" {kind}" if kind else "")
        return text + (f": {message}" if message else "")

    def _redact(self, text: str) -> str:
        text = "".join(c if c.isprintable() and c.isascii() else " " for c in text)
        text = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", text)
        token = self._token or ""
        if token:
            text = text.replace(token, "[redacted]")
            words = re.findall(r"\S{12,}", text)
            for word in words:
                if word in token or token in word:
                    text = text.replace(word, "[redacted]")
        return text[:MAX_ERROR_CHARS]

    def _answer(self, raw: bytes) -> Answer:
        data = _dict(_load_json(raw), "the reply")
        stop = data.get("stopReason")
        if not isinstance(stop, str):
            raise JudgeError("Bedrock sent a malformed reply (no stop reason)")
        usage = data.get("usage")
        tokens = (
            (_count(usage.get("inputTokens")), _count(usage.get("outputTokens")))
            if isinstance(usage, dict)
            else (None, None)
        )
        if tokens[0] is None or tokens[1] is None:
            self.usage.missing += 1
        else:
            self.usage.input_tokens += tokens[0]
            self.usage.output_tokens += tokens[1]
        if stop in FILTERED_STOPS:
            return "unsure"
        message = _dict(_dict(data.get("output"), "output").get("message"), "the message")
        content = message.get("content")
        if not isinstance(content, list):
            raise JudgeError("Bedrock sent a malformed reply (no content)")
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return parse_answer("".join(texts))


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

# (truth, image), or (truth, image, the item's height_px), or from the gold set (truth,
# image, height_px, whether the item is held out: outside the screening subset).
Crop = (
    tuple[goldset.Label, Image]
    | tuple[goldset.Label, Image, int]
    | tuple[goldset.Label, Image, int, bool]
)
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
    held_out: list[bool] = field(default_factory=list)  # per answer: outside the screen
    usage: Usage | None = None  # a hosted model's requests and tokens


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
    usage = getattr(judge, "usage", None)
    run.usage = usage if isinstance(usage, Usage) else None
    try:
        for crop in crops():
            truth, image = crop[0], crop[1]
            t0 = time.perf_counter()
            answer = judge.classify(image)
            run.seconds += time.perf_counter() - t0
            run.answers.append((truth, answer))
            run.heights.append(crop[2] if len(crop) >= 3 else 0)
            run.held_out.append(crop[3] if len(crop) == 4 else False)
    except (JudgeError, goldset.GoldsetError, ValueError) as exc:
        run.error = str(exc)
    finally:
        judge.close()
    run.peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return run


def held_out(manifest: goldset.Manifest) -> tuple[goldset.Item, ...]:
    """The gold-set items outside the screening subset: never used to tune the prompt."""
    screen = {item.id for item in goldset.screening_subset(manifest)}
    return tuple(item for item in manifest.items if item.id not in screen)


def _gold_crops(subset: str, limit: int | None) -> Crops:
    def crops() -> Iterator[Crop]:
        manifest = goldset.load_manifest()
        held = {item.id for item in held_out(manifest)}
        items = goldset.screening_subset(manifest) if subset == "screen" else manifest.items
        for item, image in goldset.iter_gold(manifest, items=items[:limit]):
            # A crop of the licensed gold set: it may be sent to a hosted model.
            yield item.label, mark_licensed(image), item.height_px, item.id in held

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


def _title(name: str) -> str:
    if name in HOSTED:
        h = HOSTED[name]
        return f"{h.name} ({h.family}, {h.model_id}, {h.region}, {h.licence})"
    c = CANDIDATES[name]
    return f"{c.name} ({c.family}, {c.params_b:g}B parameters, {c.licence})"


def _cost_per_crop(run: Run) -> float:
    """Measured cost per crop answered, in dollars; inf when it cannot be measured."""
    if run.name not in HOSTED or run.usage is None or run.usage.missing or not run.answers:
        return math.inf
    usage = run.usage
    return cost_usd(HOSTED[run.name], usage.input_tokens, usage.output_tokens) / len(run.answers)


def _usage(run: Run) -> list[str]:
    """A hosted model's region, requests, tokens and measured cost."""
    if run.name not in HOSTED or run.usage is None:
        return []
    h, u = HOSTED[run.name], run.usage
    cost = cost_usd(h, u.input_tokens, u.output_tokens)
    per_crop = cost / len(run.answers) if run.answers else math.nan
    per_bar = "n/a" if math.isnan(per_crop) else f"${BAR_CROPS * per_crop:.4f}"
    lines = [
        f"region {h.region}, requests {u.requests}, input tokens {u.input_tokens}, "
        f"output tokens {u.output_tokens}",
        f"cost ${cost:.4f} at ${h.input_usd_per_mtok:g} and ${h.output_usd_per_mtok:g} per "
        f"million input and output tokens, cost per {BAR_CROPS} crops {per_bar}",
    ]
    if u.missing:
        lines.append(f"usage missing from {u.missing} replies: the cost is a lower bound")
    return lines


def _held_out(run: Run) -> list[tuple[goldset.Label, Answer]]:
    if len(run.held_out) != len(run.answers):
        return []
    return [pair for pair, held in zip(run.answers, run.held_out, strict=True) if held]


def bakeoff(
    names: Sequence[str],
    run_one: Callable[[str], Run],
    emit: Callable[[str], None] = print,
    *,
    decide: bool = True,
) -> None:
    """Run each candidate, report it as soon as it is done (the whole set, then the
    held-out items), then the pairs, then the verdict. A model passes only if the whole
    set and the held-out items both pass the quality bar; `decide` is False for a screen
    or a limited run, which decides nothing."""
    runs: list[Run] = []
    passing: list[Run] = []
    for name in names:
        run = run_one(name)
        runs.append(run)
        title = _title(run.name)
        if run.error is not None:
            head = f"stopped after {len(run.answers)} crops" if run.answers else "not run"
            for line in [f"== {title}", f"{head}: {run.error}", *_usage(run)]:
                emit(line)
            continue
        scores = score(run.answers)
        extra = ""
        if run.name not in HOSTED:
            extra = f", load {run.load_seconds:.1f} s, peak RAM {run.peak_rss_mib / 1024:.2f} GiB"
        for line in report(title, scores, run.seconds, extra) + _usage(run) + _by_height(run):
            emit(line)
        held = _held_out(run)
        per_crop = run.seconds / scores.n if scores.n else math.inf
        if held:
            held_scores = score(held)
            for line in report(
                f"{run.name}, held out (not in the screening subset)",
                held_scores,
                per_crop * len(held),
            ):
                emit(line)
            if scores.n and passes(scores, per_crop) and passes(held_scores, per_crop):
                passing.append(run)
    complete = [r for r in runs if r.error is None and r.answers]
    for a, b in itertools.combinations(complete, 2):
        if [t for t, _ in a.answers] != [t for t, _ in b.answers]:
            continue
        pairs = [(t, agree(x, y)) for (t, x), (_, y) in zip(a.answers, b.answers, strict=True)]
        for line in report(
            f"agreement of {a.name} and {b.name}", score(pairs), a.seconds + b.seconds
        ):
            emit(line)
    metered = [r.usage for r in runs if r.usage is not None]
    if metered:
        total = sum(
            cost_usd(HOSTED[r.name], r.usage.input_tokens, r.usage.output_tokens)
            for r in runs
            if r.usage is not None and r.name in HOSTED
        )
        emit(
            f"requests made {sum(u.requests for u in metered)}, "
            f"input tokens {sum(u.input_tokens for u in metered)}, "
            f"output tokens {sum(u.output_tokens for u in metered)}, cost ${total:.4f}"
        )
    hosted = [r for r in passing if r.name in HOSTED]
    local = [r for r in passing if r.name not in HOSTED]
    if not decide:
        emit("a screen or a limited run: no pass decision (run --subset all without --limit)")
    elif not passing:
        emit("no single model passes the quality bar")
    if decide and hosted:
        cheapest = min(hosted, key=lambda r: (_cost_per_crop(r), r.name))
        per_crop = _cost_per_crop(cheapest)
        cost = "n/a" if math.isinf(per_crop) else f"${BAR_CROPS * per_crop:.4f}"
        emit(f"cheapest model that passes: {cheapest.name} ({cost} per {BAR_CROPS} crops)")
    if decide and local:
        best = max(local, key=lambda r: (score(r.answers).accuracy, r.name))
        emit(f"most accurate model that passes: {best.name}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m wearreport.tools.judge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bakeoff", action="store_true", help="run the candidates on the gold set")
    ap.add_argument(
        "--backend",
        choices=("local", "bedrock"),
        default="local",
        help="local: CANDIDATES on this CPU; bedrock: HOSTED, through Amazon Bedrock",
    )
    ap.add_argument(
        "--models",
        default=None,
        help="comma-separated names (default: every candidate of the backend but the "
        "backups); chosen = CHOSEN (refused: none is chosen)",
    )
    ap.add_argument("--subset", choices=("all", "screen"), default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N crops only")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="required with --backend bedrock: stop after N requests in all (retries count)",
    )
    return ap


def main(
    argv: Sequence[str] | None = None,
    *,
    open_judge: Callable[[Candidate], Classifier] | None = None,
    crops: Crops | None = None,
    endpoint: str | None = None,
) -> int:
    """Run the bake-off. `open_judge` and `crops` replace the real local models and gold
    set, and `endpoint` the Bedrock origin (tests); they run in this process."""
    args = build_parser().parse_args(argv)
    if not args.bakeoff:
        build_parser().print_help()
        return 2
    remote = args.backend == "bedrock"
    registry: dict[str, Candidate] | dict[str, HostedCandidate] = HOSTED if remote else CANDIDATES
    if args.models is None:
        names = [n for n, c in HOSTED.items() if not c.backup] if remote else list(CANDIDATES)
    else:
        names = [n.strip() for n in args.models.split(",") if n.strip()]
    if "chosen" in names and CHOSEN is None:
        print(f"judge: {NO_CHOSEN_MODEL}", file=sys.stderr)
        return 2
    names = [CHOSEN if n == "chosen" and CHOSEN is not None else n for n in names]
    unknown = [n for n in names if n not in registry]
    if unknown or not names:
        print(f"judge: unknown candidate {unknown[0] if unknown else ''!r}", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("judge: --limit must be at least 1", file=sys.stderr)
        return 2
    decide = crops is not None or (args.subset == "all" and args.limit is None)

    run_one: Callable[[str], Run]
    if remote:
        if args.max_requests is None or args.max_requests < 1:
            print("judge: a remote run needs --max-requests N (N >= 1)", file=sys.stderr)
            return 2
        if not os.environ.get(TOKEN_ENV):
            print(f"judge: {TOKEN_ENV} is not set", file=sys.stderr)
            return 2
        budget = RequestBudget(args.max_requests)
        remote_source = crops or _gold_crops(args.subset, args.limit)

        def run_one(name: str) -> Run:
            return evaluate(
                name,
                lambda: BedrockClassifier(HOSTED[name], budget=budget, endpoint=endpoint),
                remote_source,
            )
    elif open_judge is not None or crops is not None:
        opener = open_judge or (lambda c: Judge(c, threads=args.threads))
        source = crops or _gold_crops(args.subset, args.limit)

        def run_one(name: str) -> Run:
            return evaluate(name, lambda: opener(CANDIDATES[name]), source)
    else:
        context = multiprocessing.get_context("spawn")

        def run_one(name: str) -> Run:
            with context.Pool(1) as pool:
                return pool.apply(_child, (name, args.subset, args.limit, args.threads))

    bakeoff(names, run_one, lambda line: print(line, flush=True), decide=decide)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
