"""The hosted judge's HTTP path: DeepInfra's vision models, called with the standard
library only, and what every hosted backend shares.

This module is what the spot-check tool needs of the judge, and nothing more: it imports
neither llama-cpp-python, the POSIX-only `resource` module nor `wearreport.tools.judge`, so
it imports on Windows. `wearreport.tools.judge` (the local models, Amazon Bedrock and the
bake-off) imports every name here back, so each `judge.X` still resolves to the same object.

Each request carries one crop, as a PNG built in memory and sent as a base64 data URL, and
asks PROMPT; the reply is parsed by `parse_answer` into person, in_vehicle, not_person or
unsure. Every request has a timeout, and is counted against a shared `RequestBudget`
(retries count); throttling and server errors are retried at most MAX_RETRIES times with
backoff, other errors not at all; redirects are never followed; replies are capped and
parsed as hostile input; an error never carries a credential.

Three entry points, and only three, send crops (AGENTS.md INV-1):

- `DeepInfraClassifier.classify`, the bake-off's: gold-set and control crops only
  (`mark_licensed`); anything else is refused before a request is made. It sends no
  Authorization header: the environment adds the credential.
- `LiveCropJudge.classify_live_crop`, the paired spot-check's, and nothing else's: one
  detection crop of a live camera frame per request, to the pinned DeepInfra origin
  (DEEPINFRA_ORIGIN; a loopback server in tests). The key comes from DEEPINFRA_API_KEY
  when it is set, and is sent as a bearer token; without it no Authorization header is
  sent, for an environment that injects the credential. Nothing about a crop is logged,
  cached or written.
- `LiveAttributeJudge.classify_attributes`, the attribute session's, and nothing else's:
  the same transport, origin, key handling, request budget, retries, timeouts and error
  redaction as LiveCropJudge, but it asks ATTRIBUTE_PROMPT, three yes/no questions about
  the person in the crop, and parses the reply with `parse_attributes`.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt

from wearreport._cv import cv2

Answer = Literal["person", "in_vehicle", "not_person", "unsure"]
ANSWERS: tuple[Answer, ...] = get_args(Answer)
Image = npt.NDArray[np.uint8]

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
MAX_IMAGE_SIDE = 2048

# The attribute questions, and the one line they must be answered with: yes, no or unsure
# for each, in this order. The answers are returned as three letters, y, n or u, in the
# same order (e.g. "ynu").
ATTRIBUTE_PROMPT = (
    "This picture is a crop from a low-resolution street camera. "
    "Look at the person inside the green box.\n"
    "outer - is the person wearing an outer layer (a coat or a jacket)?\n"
    "legs - are the person's legs bare (shorts or a short skirt)?\n"
    "umbrella - is the person holding an open umbrella?\n"
    "Answer each with yes, no or unsure (unsure when you cannot tell). Reply with exactly "
    "one line in this form and nothing else, for example:\n"
    "outer=yes legs=no umbrella=unsure"
)
ATTRIBUTE_VALUES = {"yes": "y", "no": "n", "unsure": "u"}
_ATTRIBUTE_REPLY = re.compile(
    r"outer=(yes|no|unsure) legs=(yes|no|unsure) umbrella=(yes|no|unsure)"
)
MAX_ATTRIBUTE_REPLY_CHARS = 64
ALL_UNSURE = "uuu"


class JudgeError(RuntimeError):
    """The model cannot be opened or failed on an image."""


# Candidates -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HostedCandidate:
    """One vision model served by a hosted provider.

    Amazon Bedrock (HOSTED) is called with the Converse API. The model ID, the regions and
    the terms are from the model's page in the Bedrock documentation (`card`); the prices,
    in US dollars per million tokens on demand (Standard tier) in `region`, from the Amazon
    Bedrock pricing page. A model ID that starts with eu. or us. is a cross-region
    inference profile: Bedrock serves it within that geography.

    DeepInfra (DEEPINFRA) is called with its OpenAI-compatible chat completions API. The
    model ID, the prices (standard tier) and the licence are from the model's DeepInfra
    page (`card`) and its entry in DeepInfra's model list."""

    name: str
    family: str
    model_id: str
    region: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    licence: str  # of the weights, or the terms under which the provider serves the model
    card: str
    backup: bool = False  # run only when no other hosted candidate passes
    reasoning_off: bool = False  # DeepInfra: ask for no reasoning (reasoning_effort none)


_DEEPINFRA_PAGE = "https://deepinfra.com/"
DEEPINFRA_REGION = "deepinfra"  # DeepInfra does not let a caller choose a region
# Prices from DeepInfra's model list on 2026-09-26 (cents per token, times 10^4).
DEEPINFRA: dict[str, HostedCandidate] = {
    c.name: c
    for c in (
        HostedCandidate(
            name="di-qwen3-vl-235b",
            family="Qwen3-VL",
            model_id="Qwen/Qwen3-VL-235B-A22B-Instruct",  # FP8
            region=DEEPINFRA_REGION,
            input_usd_per_mtok=0.20,
            output_usd_per_mtok=0.88,
            licence="Apache-2.0",
            card=_DEEPINFRA_PAGE + "Qwen/Qwen3-VL-235B-A22B-Instruct",
        ),
        HostedCandidate(
            name="di-qwen3.5-397b",
            family="Qwen3.5",
            model_id="Qwen/Qwen3.5-397B-A17B",  # FP8; reasons unless told not to
            region=DEEPINFRA_REGION,
            input_usd_per_mtok=0.45,
            output_usd_per_mtok=3.00,
            licence="Apache-2.0",
            card=_DEEPINFRA_PAGE + "Qwen/Qwen3.5-397B-A17B",
            reasoning_off=True,
        ),
        HostedCandidate(
            name="di-llama4-maverick",
            family="Meta Llama 4",
            model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  # retires 2026-10-01
            region=DEEPINFRA_REGION,
            input_usd_per_mtok=0.20,
            output_usd_per_mtok=0.80,
            licence="Llama 4 Community License",
            card=_DEEPINFRA_PAGE + "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        ),
        HostedCandidate(
            name="di-kimi-k2.6",
            family="Moonshot Kimi",
            model_id="moonshotai/Kimi-K2.6",  # FP4; reasons unless told not to
            region=DEEPINFRA_REGION,
            input_usd_per_mtok=0.75,
            output_usd_per_mtok=3.50,
            licence="Modified MIT License",
            card=_DEEPINFRA_PAGE + "moonshotai/Kimi-K2.6",
            reasoning_off=True,
        ),
        HostedCandidate(
            name="di-kimi-k3",
            family="Moonshot Kimi",
            model_id="moonshotai/Kimi-K3",  # reasons in its reply unless told not to
            region=DEEPINFRA_REGION,
            input_usd_per_mtok=2.85,
            output_usd_per_mtok=14.25,
            licence="Kimi K3 License (MIT-style; a separate agreement for large model services)",
            card=_DEEPINFRA_PAGE + "moonshotai/Kimi-K3",
            reasoning_off=True,
        ),
        HostedCandidate(
            name="di-glm-4.6v",
            family="Z.ai GLM-V",
            model_id="zai-org/GLM-4.6V",  # listed as deprecated, still served
            region=DEEPINFRA_REGION,
            input_usd_per_mtok=0.30,
            output_usd_per_mtok=0.90,
            licence="MIT",
            card=_DEEPINFRA_PAGE + "zai-org/GLM-4.6V",
        ),
        HostedCandidate(
            name="di-gemma-4-31b",
            family="Google Gemma 4",
            model_id="google/gemma-4-31B-it",  # FP8
            region=DEEPINFRA_REGION,
            input_usd_per_mtok=0.13,
            output_usd_per_mtok=0.38,
            licence="Apache-2.0",
            card=_DEEPINFRA_PAGE + "google/gemma-4-31B-it",
        ),
    )
}


def cost_usd(candidate: HostedCandidate, input_tokens: int, output_tokens: int) -> float:
    """What `input_tokens` and `output_tokens` cost at the candidate's published prices."""
    return (
        input_tokens * candidate.input_usd_per_mtok + output_tokens * candidate.output_usd_per_mtok
    ) / 1e6


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


def parse_attributes(text: object) -> str:
    """The three answers of a reply to ATTRIBUTE_PROMPT, as letters (e.g. "ynu"). The
    reply must be exactly the one line the prompt asks for, give or take surrounding
    white space, within MAX_ATTRIBUTE_REPLY_CHARS; anything else, including text before or
    after it, another order or an empty reply, is ALL_UNSURE."""
    if not isinstance(text, str) or len(text) > MAX_ATTRIBUTE_REPLY_CHARS:
        return ALL_UNSURE
    match = _ATTRIBUTE_REPLY.fullmatch(text.strip())
    if match is None:
        return ALL_UNSURE
    return "".join(ATTRIBUTE_VALUES[word] for word in match.groups())


def validate_image(image: object) -> Image:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("an image must be a uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"an image must be HxWx3 (BGR), not shape {image.shape}")
    height, width = image.shape[:2]
    if not (0 < height <= MAX_IMAGE_SIDE and 0 < width <= MAX_IMAGE_SIDE):
        raise ValueError(f"an image must be 1 to {MAX_IMAGE_SIDE} pixels on each side")
    return image


# Crops that may be sent to a hosted model -------------------------------------------------

# The bake-off sends only crops of the licensed gold set and the committed controls. The
# code that renders them marks each one; `classify` refuses anything unmarked, a copy, or a
# marked array whose pixels have changed since. Live crops go only through LiveCropJudge.
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


# The hosted backends ----------------------------------------------------------------------

REQUEST_TIMEOUT = 60.0  # seconds, for each request in all (each retry has its own)
MAX_RETRIES = 3  # after the first attempt, on throttling and server errors only
BACKOFF_SECONDS = 2.0  # doubled after each retry
REMOTE_MAX_TOKENS = 10
ATTRIBUTE_MAX_TOKENS = 32  # the longest well-formed attribute line is about 15 tokens
PNG_SUFFIX = ".png"  # the format each crop is sent in, encoded in memory
MAX_RESPONSE_BYTES = 1 << 20
READ_CHUNK_BYTES = 1 << 16
MAX_ERROR_CHARS = 200
# Refusals of the credential: their messages can echo a key in any form, so only the status
# and the error kind are reported.
AUTH_STATUSES = frozenset({401, 403, 407})
DEEPINFRA_ORIGIN = "https://api.deepinfra.com"
DEEPINFRA_PATH = "/v1/openai/chat/completions"
DEEPINFRA_FILTERED = frozenset({"content_filter"})
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})  # a local fake server (tests)
_DEEPINFRA_MODEL_ID = re.compile(r"[A-Za-z0-9][\w.-]{0,63}/[A-Za-z0-9][\w.-]{0,127}")
_TOKEN = re.compile(r"[\x21-\x7e]{1,8192}")
# A provider's error message is reduced to plain words (an allowlist, not a blocklist of
# key shapes): a word is kept only if it is at most 15 ASCII letters, or one of the service
# error names below, with at most one trailing punctuation mark. Every other word, which
# may be an echoed key in any form, becomes "…". Only the first 4 * MAX_ERROR_CHARS
# characters of a message are examined, so a hostile reply costs linear time at most, and a
# word split by that cut is dropped.
_PLAIN_WORD = re.compile(r"([A-Za-z]+)([.,:;!?]?)")
MAX_WORD_LETTERS = 15
KNOWN_ERROR_NAMES = frozenset(
    {
        "AccessDeniedException",
        "ConflictException",
        "InternalServerException",
        "ModelErrorException",
        "ModelNotReadyException",
        "ModelStreamErrorException",
        "ModelTimeoutException",
        "ResourceNotFoundException",
        "ServiceQuotaExceededException",
        "ServiceUnavailableException",
        "ThrottlingException",
        "UnrecognizedClientException",
        "ValidationException",
    }
)


def _cut_words(text: str, limit: int) -> str:
    """The first `limit` characters of `text`, with a word that the cut splits replaced by
    "…": its first part could be the first letters of a key and pass as a plain word."""
    cut = text[:limit]
    if len(text) > limit and not text[limit].isspace() and not cut[-1:].isspace():
        cut = cut[: len(cut) - len(cut.split()[-1])] + "…"
    return cut


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
    """Never follow a redirect: it would carry a credential to another address."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _origin(base: str, provider: str) -> tuple[str, bool]:
    """`base` checked to be a bare https origin (or http on this machine), and whether it
    is on this machine."""
    parts = urllib.parse.urlsplit(base)
    local = parts.hostname in LOOPBACK
    if parts.scheme != "https" and not (parts.scheme == "http" and local):
        raise JudgeError(f"the {provider} endpoint must be an https URL")
    if parts.path.strip("/") or parts.query or parts.fragment or parts.username:
        raise JudgeError(f"the {provider} endpoint must be a bare origin")
    return f"{parts.scheme}://{parts.netloc}", local


class _DeadlinePassed(JudgeError):
    """A request ran past its deadline while its reply was being read."""


def _set_read_timeout(response: Any, seconds: float) -> None:
    """Make the next socket read under `response` wait at most `seconds`, so that no single
    read outlasts the request's deadline. Does nothing if the socket cannot be found."""
    layer = response
    for _ in range(4):  # HTTPError -> HTTPResponse -> BufferedReader -> SocketIO
        sock = getattr(layer, "_sock", None)
        if isinstance(sock, socket.socket):
            sock.settimeout(seconds)
            return
        layer = getattr(layer, "raw", None) or getattr(layer, "fp", None)


class _Watchdog:
    """Shuts a request's socket down when the request's deadline passes. http.client reads
    the status line, the headers, chunk-size lines and trailers without returning to
    _read_capped, so only closing the socket under it bounds every read in time."""

    def __init__(self, seconds: float) -> None:
        self.fired = False
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._timer = threading.Timer(seconds, self._fire)
        self._timer.daemon = True

    def start(self) -> None:
        self._timer.start()

    def cancel(self) -> None:
        self._timer.cancel()

    def attach(self, sock: socket.socket | None) -> None:
        """Watch `sock`, the request's connected socket; shut it at once if already late."""
        with self._lock:
            self._sock = sock
            fired = self.fired
        if fired:
            _shut(sock)

    def _fire(self) -> None:
        with self._lock:
            self.fired = True
            sock = self._sock
        _shut(sock)


def _shut(sock: socket.socket | None) -> None:
    """End every read and write on `sock`, from any thread. The plain socket's shutdown is
    used even for a TLS socket, so the TLS state is left to the thread reading it."""
    if sock is None:
        return
    with contextlib.suppress(OSError):  # already closed
        socket.socket.shutdown(sock, socket.SHUT_RDWR)


class _WatchedHTTPConnection(http.client.HTTPConnection):
    watchdog: _Watchdog | None = None

    def connect(self) -> None:
        super().connect()
        if self.watchdog is not None:
            self.watchdog.attach(self.sock)


class _WatchedHTTPSConnection(http.client.HTTPSConnection):
    watchdog: _Watchdog | None = None

    def connect(self) -> None:
        super().connect()
        if self.watchdog is not None:
            self.watchdog.attach(self.sock)


def _watched(
    cls: type[_WatchedHTTPConnection] | type[_WatchedHTTPSConnection],
    watchdog: Callable[[], _Watchdog | None],
) -> Callable[..., http.client.HTTPConnection]:
    """A connection factory for urllib that hands each connection the current watchdog."""

    def connection(*args: Any, **kwargs: Any) -> http.client.HTTPConnection:
        made = cls(*args, **kwargs)
        made.watchdog = watchdog()
        return made

    return connection


class _WatchedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, watchdog: Callable[[], _Watchdog | None]) -> None:
        super().__init__()
        self._watchdog = watchdog

    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_watched(_WatchedHTTPConnection, self._watchdog), req=req)


class _WatchedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, watchdog: Callable[[], _Watchdog | None]) -> None:
        super().__init__(context=ssl.create_default_context())
        self._watchdog = watchdog

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(
            _watched(_WatchedHTTPSConnection, self._watchdog),
            req=req,
            context=self._context,  # type: ignore[attr-defined]
        )


def _read_capped(response: Any, provider: str, deadline: float) -> bytes:
    """The reply body, read in chunks: at most MAX_RESPONSE_BYTES, and all of it before
    `deadline` (time.monotonic()), so a server that sends slowly cannot hold a request."""
    chunks: list[bytes] = []
    size = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _DeadlinePassed(f"{provider} took too long to send its reply")
        _set_read_timeout(response, remaining)
        chunk = response.read1(min(READ_CHUNK_BYTES, MAX_RESPONSE_BYTES + 1 - size))
        if not isinstance(chunk, bytes):
            raise JudgeError(f"{provider} sent a malformed reply")
        if not chunk:  # the end, or the watchdog shut the socket
            if time.monotonic() >= deadline:
                raise _DeadlinePassed(f"{provider} took too long to send its reply")
            return b"".join(chunks)
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise JudgeError(f"{provider} sent a reply larger than {MAX_RESPONSE_BYTES} bytes")


def _reject_constant(name: str) -> object:
    raise ValueError(f"{name} is not a number")


def _load_json(raw: bytes, provider: str = "Bedrock") -> object:
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError, OverflowError) as exc:
        raise JudgeError(f"{provider} sent a malformed reply ({type(exc).__name__})") from None


def _dict(value: object, what: str, provider: str = "Bedrock") -> dict[str, object]:
    if not isinstance(value, dict):
        raise JudgeError(f"{provider} sent a malformed reply ({what} is not an object)")
    return value


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 10**9 else None


def _png_base64(pixels: Image) -> str:
    ok, encoded = cv2.imencode(PNG_SUFFIX, pixels)  # in memory
    if not ok:
        raise JudgeError("cannot encode the crop")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class _HostedClassifier:
    """What every hosted backend shares: one crop per request, only marked gold-set and
    control crops, a shared request budget, a timeout on every request, at most
    MAX_RETRIES retries on throttling and server errors, no redirects, capped replies
    parsed as hostile input, and errors that never carry a credential. `endpoint` replaces
    the provider's origin (tests: a fake server on the loopback interface); `sleep` waits
    between retries."""

    provider = "hosted"

    def __init__(
        self,
        candidate: HostedCandidate,
        *,
        budget: RequestBudget,
        url: str,
        local: bool,
        token: str | None,
        timeout: float,
        sleep: Callable[[float], None],
    ) -> None:
        if not 0 < timeout <= 600:
            raise ValueError("timeout must be from 0 to 600 seconds")
        self.candidate = candidate
        self.usage = Usage()
        self._budget = budget
        self._timeout = timeout
        self._sleep = sleep
        self._url = url
        self._token = token  # sent as a bearer token when not None
        self._closed = False
        self._watchdog: _Watchdog | None = None  # the current request's
        # The system's proxy settings apply, except to a server on this machine.
        proxies = urllib.request.ProxyHandler({} if local else None)
        self._opener = urllib.request.build_opener(
            proxies,
            _NoRedirect(),
            _WatchedHTTPHandler(lambda: self._watchdog),
            _WatchedHTTPSHandler(lambda: self._watchdog),
        )

    def classify(self, image: Image) -> Answer:
        """The model's answer about the box drawn in `image`, a marked gold-set or control
        crop (HxWx3 uint8, BGR)."""
        if self._closed:
            raise JudgeError("the judge is closed")
        pixels = validate_image(image)
        if not is_licensed(pixels):
            raise JudgeError("refusing an image that is not a gold-set or control crop")
        return self._ask(pixels)

    def _ask(self, pixels: Image) -> Answer:
        """One crop, already checked by the entry point that took it, sent and answered."""
        if self._closed:
            raise JudgeError("the judge is closed")
        return self._answer(self._call(self._body(pixels)))

    def close(self) -> None:
        self._closed = True
        self._token = None

    def _body(self, pixels: Image) -> bytes:
        raise NotImplementedError

    def _answer(self, raw: bytes) -> Answer:
        raise NotImplementedError

    def _error_kind(self, exc: urllib.error.HTTPError, raw: bytes) -> str:
        return ""

    def _retryable(self, status: int, kind: str) -> bool:
        return status == 429 or status >= 500

    def _call(self, body: bytes) -> bytes:
        for attempt in range(MAX_RETRIES + 1):
            status, raw, kind = self._send(body)
            if 200 <= status < 300:
                return raw
            if not self._retryable(status, kind) or attempt == MAX_RETRIES:
                raise JudgeError(self._describe(status, kind, raw))
            self._sleep(BACKOFF_SECONDS * 2**attempt)
        raise AssertionError("unreachable")

    def _send(self, body: bytes) -> tuple[int, bytes, str]:
        """One request: the HTTP status, the body and the provider's error type."""
        if self._closed:
            raise JudgeError("the judge is closed")
        self._budget.take()
        self.usage.requests += 1
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(  # noqa: S310  (the scheme is checked above)
            self._url, data=body, method="POST", headers=headers
        )
        timed_out = f"the request timed out after {self._timeout:g} s"
        deadline = time.monotonic() + self._timeout  # for the whole request
        watchdog = _Watchdog(self._timeout)
        self._watchdog = watchdog
        watchdog.start()
        try:
            status, raw, kind = self._exchange(request, deadline)
        except (TimeoutError, _DeadlinePassed):
            raise JudgeError(timed_out) from None
        except urllib.error.URLError as exc:
            if watchdog.fired or isinstance(exc.reason, TimeoutError):
                raise JudgeError(timed_out) from None
            raise JudgeError(
                f"cannot reach {self.provider} ({type(exc.reason).__name__})"
            ) from None
        except (OSError, http.client.HTTPException, ValueError) as exc:
            if watchdog.fired:  # the socket was shut under a read: a reset, IncompleteRead
                raise JudgeError(timed_out) from None
            raise JudgeError(f"the request failed ({type(exc).__name__})") from None
        finally:
            watchdog.cancel()
            self._watchdog = None
        if watchdog.fired or time.monotonic() >= deadline:
            raise JudgeError(timed_out)
        return status, raw, kind

    def _exchange(self, request: urllib.request.Request, deadline: float) -> tuple[int, bytes, str]:
        """Send `request` and read its reply: the HTTP status, the body and the error type."""
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                return response.status, _read_capped(response, self.provider, deadline), ""
        except urllib.error.HTTPError as exc:
            try:
                raw = _read_capped(exc, self.provider, deadline)
            except (JudgeError, OSError, http.client.HTTPException, ValueError):
                raw = b""
            finally:
                exc.close()
            return exc.code, raw, self._error_kind(exc, raw)

    def _message(self, data: object) -> str:
        """The error message in a provider's error body, or ''."""
        if isinstance(data, dict):
            found = data.get("message") or data.get("Message")
            return found if isinstance(found, str) else ""
        return ""

    def _describe(self, status: int, kind: str, raw: bytes) -> str:
        """An error for a failed request, with the service's message, never a credential.
        A refusal of the credential (AUTH_STATUSES) carries no message at all."""
        text = f"{self.provider} answered HTTP {status}" + (f" {kind}" if kind else "")
        if status in AUTH_STATUSES:
            return text
        try:
            message = self._message(json.loads(raw.decode("utf-8")))
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError, OverflowError):
            message = ""
        message = self._plain_words(message)
        return text + (f": {message}" if message else "")

    def _plain_words(self, text: str) -> str:
        """`text` with every word that is not plain (see _PLAIN_WORD) replaced by one "…", and
        the word after "Bearer" always replaced. The known token is removed first, and a
        word of 8 or more letters found in it is not kept."""
        text = _cut_words(text, 4 * MAX_ERROR_CHARS)
        token = self._token or ""
        if token:
            text = text.replace(token, " ")
        words: list[str] = []
        after_bearer = False
        for word in text.split():
            match = _PLAIN_WORD.fullmatch(word)
            letters = match[1] if match else ""
            keep = (
                match is not None
                and not after_bearer
                and (len(letters) <= MAX_WORD_LETTERS or letters in KNOWN_ERROR_NAMES)
                and not (token and len(letters) >= 8 and letters in token)
            )
            after_bearer = letters.lower() == "bearer"
            if keep:
                words.append(word)
            elif not words or words[-1] != "…":
                words.append("…")
        return " ".join(words)[:MAX_ERROR_CHARS]

    def _add_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        if input_tokens is None or output_tokens is None:
            self.usage.missing += 1
        else:
            self.usage.input_tokens += input_tokens
            self.usage.output_tokens += output_tokens


class DeepInfraClassifier(_HostedClassifier):
    """A DeepInfra candidate answering PROMPT, one crop per chat completions request (the
    OpenAI-compatible API), at temperature 0, the crop as a base64 PNG data URL built in
    memory. It sends no Authorization header of its own: the environment adds one to
    requests for api.deepinfra.com (the session's network proxy), so no key is read,
    held or sent by this code."""

    provider = "DeepInfra"
    prompt = PROMPT
    max_tokens = REMOTE_MAX_TOKENS

    def __init__(
        self,
        candidate: HostedCandidate,
        *,
        budget: RequestBudget,
        endpoint: str | None = None,
        timeout: float = REQUEST_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        origin, local = _origin(endpoint or DEEPINFRA_ORIGIN, self.provider)
        if not _DEEPINFRA_MODEL_ID.fullmatch(candidate.model_id):
            raise JudgeError("malformed model ID")
        super().__init__(
            candidate,
            budget=budget,
            url=origin + DEEPINFRA_PATH,
            local=local,
            token=None,
            timeout=timeout,
            sleep=sleep,
        )

    def _body(self, pixels: Image) -> bytes:
        image = "data:image/png;base64," + _png_base64(pixels)
        request: dict[str, object] = {
            "model": self.candidate.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "stream": False,
        }
        if self.candidate.reasoning_off:
            request["reasoning_effort"] = "none"
        return json.dumps(request).encode()

    def _message(self, data: object) -> str:
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return super()._message(error)
            for key in ("detail", "error", "message"):
                if isinstance(data.get(key), str):
                    return str(data[key])
        return ""

    def _answer(self, raw: bytes) -> Answer:
        text = self._reply_text(raw)
        return "unsure" if text is None else parse_answer(text)

    def _reply_text(self, raw: bytes) -> str | None:
        """The reply's text, or None when the model said nothing (a refusal, a filtered
        reply, or reasoning that ran out of tokens). Counts the tokens used."""
        p = self.provider
        data = _dict(_load_json(raw, p), "the reply", p)
        usage = data.get("usage")
        if isinstance(usage, dict):
            self._add_usage(
                _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))
            )
        else:
            self._add_usage(None, None)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise JudgeError(f"{p} sent a malformed reply (no choices)")
        choice = _dict(choices[0], "a choice", p)
        finish = choice.get("finish_reason")
        if finish is not None and not isinstance(finish, str):
            raise JudgeError(f"{p} sent a malformed reply (the finish reason is not text)")
        if finish in DEEPINFRA_FILTERED:
            return None
        message = _dict(choice.get("message"), "the message", p)
        content = message.get("content")
        if content is None:
            return None  # nothing said (a refusal, or reasoning that ran out of tokens)
        if not isinstance(content, str):
            raise JudgeError(f"{p} sent a malformed reply (the content is not text)")
        return content


# Live camera crops (the paired spot-check) ------------------------------------------------

API_KEY_ENV = "DEEPINFRA_API_KEY"  # the variable name, not a key


class _LiveDeepInfra(DeepInfraClassifier):
    """DeepInfraClassifier with the key of a local run: sent as a bearer token when set.
    Its errors never carry the provider's message text, which a hostile reply could fill
    with pieces of the key."""

    def __init__(self, candidate: HostedCandidate, *, key: str | None, **kwargs: Any) -> None:
        super().__init__(candidate, **kwargs)
        self._token = key

    def _describe(self, status: int, kind: str, raw: bytes) -> str:
        """The HTTP status and the error kind only."""
        return f"{self.provider} answered HTTP {status}" + (f" {kind}" if kind else "")


class _LiveAttributes(_LiveDeepInfra):
    """_LiveDeepInfra asking ATTRIBUTE_PROMPT instead of PROMPT."""

    prompt = ATTRIBUTE_PROMPT
    max_tokens = ATTRIBUTE_MAX_TOKENS

    def ask_attributes(self, pixels: Image) -> str:
        """One crop, already checked by the entry point that took it, sent and answered."""
        if self._closed:
            raise JudgeError("the judge is closed")
        return parse_attributes(self._reply_text(self._call(self._body(pixels))))


def _live_options(name: str, endpoint: str | None) -> tuple[HostedCandidate, str | None]:
    """The DEEPINFRA candidate `name` and the key from API_KEY_ENV (None when unset),
    after checking that `endpoint`, if given, is DEEPINFRA_ORIGIN or a server on this
    machine. Raises JudgeError before any request."""
    candidate = DEEPINFRA.get(name)
    if candidate is None:
        raise JudgeError("unknown DeepInfra model")
    if endpoint is not None:
        origin, local = _origin(endpoint, "DeepInfra")
        if not local and origin != DEEPINFRA_ORIGIN:
            raise JudgeError(f"the DeepInfra endpoint is pinned to {DEEPINFRA_ORIGIN}")
    key: str | None = os.environ.get(API_KEY_ENV, "")
    if not key:
        key = None
    elif not _TOKEN.fullmatch(key):
        raise JudgeError(f"{API_KEY_ENV} is not a well-formed key")
    return candidate, key


class LiveCropJudge:
    """The one entry point for crops of live camera frames, used only by the paired
    spot-check: the DEEPINFRA model `name` answers PROMPT about one detection crop per
    request, as the spot-check rendered it for its reviewer (HxWx3 uint8, BGR; never a whole
    frame), encoded to PNG in memory. The endpoint is pinned to DEEPINFRA_ORIGIN; `endpoint`
    may only replace it with a server on this machine (tests). Every request counts against
    `budget`, retries included. The key is read from API_KEY_ENV once, here: when it is set
    it is sent as `Authorization: Bearer`, and it never appears in an error; when it is not
    set, no Authorization header is sent. An HTTP error names only its status, never the
    provider's message. Raises JudgeError for an unknown model, another
    endpoint or a malformed key, before any request."""

    def __init__(
        self,
        name: str,
        *,
        budget: RequestBudget,
        endpoint: str | None = None,
        timeout: float = REQUEST_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        candidate, key = _live_options(name, endpoint)
        self.candidate = candidate
        self._judge = _LiveDeepInfra(
            candidate, key=key, budget=budget, endpoint=endpoint, timeout=timeout, sleep=sleep
        )

    @property
    def url(self) -> str:
        return self._judge._url

    @property
    def usage(self) -> Usage:
        return self._judge.usage

    def classify_live_crop(self, image: Image) -> Answer:
        """The model's answer about the box drawn in `image`, one detection crop."""
        return self._judge._ask(validate_image(image))

    def close(self) -> None:
        self._judge.close()


class LiveAttributeJudge:
    """The entry point for crops of live camera frames in the attribute session, and
    nothing else: the DEEPINFRA model `name` answers ATTRIBUTE_PROMPT about one detection
    crop per request (HxWx3 uint8, BGR; never a whole frame), encoded to PNG in memory.
    It is given the pixels only. Endpoint pinning, the key, the request budget, retries,
    timeouts and error redaction are those of LiveCropJudge. Raises JudgeError for an
    unknown model, another endpoint or a malformed key, before any request."""

    def __init__(
        self,
        name: str,
        *,
        budget: RequestBudget,
        endpoint: str | None = None,
        timeout: float = REQUEST_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        candidate, key = _live_options(name, endpoint)
        self.candidate = candidate
        self._judge = _LiveAttributes(
            candidate, key=key, budget=budget, endpoint=endpoint, timeout=timeout, sleep=sleep
        )

    @property
    def url(self) -> str:
        return self._judge._url

    @property
    def usage(self) -> Usage:
        return self._judge.usage

    def classify_attributes(self, image: Image) -> str:
        """The model's three answers about the person in the box drawn in `image`, one
        detection crop, as letters: y, n or u for the outer layer, bare legs and open
        umbrella, in that order (e.g. "ynu")."""
        return self._judge.ask_attributes(validate_image(image))

    def close(self) -> None:
        self._judge.close()
