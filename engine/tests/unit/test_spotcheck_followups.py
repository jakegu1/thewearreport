"""Unit tests for the T-064 follow-ups: the failure breakdown's `other` count and the
summary parsers' duplicate-key check, on pathological inputs."""

from __future__ import annotations

from collections import Counter

import pytest

from wearreport.tools import spotcheck, spotcheck_summary


@pytest.mark.parametrize(
    ("counts", "text"),
    [
        (Counter(), "0 failed"),
        (Counter(failed=1, failed_decode=1), "1 failed (decode 1)"),
        (Counter(failed=2, failed_weird=2), "2 failed (other 2)"),
        (
            Counter(failed=4, failed_timeout=1, failed_network=2, failed_x=1),
            "4 failed (timeout 1, network 2, other 1)",
        ),
        # A kind literally named "other" is still not a known kind: counted once.
        (Counter(failed=1, failed_other=1), "1 failed (other 1)"),
    ],
)
def test_failed_text(counts: Counter[str], text: str) -> None:
    assert spotcheck._failed_text(counts) == text


PARSERS = [
    spotcheck_summary.parse_record,
    spotcheck_summary.parse_session,
    spotcheck_summary.parse_labelling,
    spotcheck_summary.parse_sweep,
]


@pytest.mark.parametrize("parse", PARSERS)
@pytest.mark.parametrize(
    "raw",
    [
        b'{"a": 1, "a": 1}',  # the same value twice is still a duplicate
        b'{"a": {"b": [[{"c": 1, "c": 2}]]}}',  # deep inside lists
        b'[{"a": 1}, {"a": 1, "a": 2}]',  # not an object at the top, a duplicate below
        b'{"\\u0061": 1, "a": 2}',  # the same key once escaped
    ],
)
def test_duplicate_keys_are_refused(parse: object, raw: bytes) -> None:
    assert callable(parse)
    with pytest.raises(ValueError, match="appears twice in one object"):
        parse(raw)


@pytest.mark.parametrize("parse", PARSERS)
def test_a_long_duplicate_key_is_not_echoed(parse: object) -> None:
    assert callable(parse)
    key = "k" * 5000
    raw = f'{{"{key}": 1, "{key}": 2}}'.encode()
    with pytest.raises(ValueError) as info:
        parse(raw)
    assert len(str(info.value)) < 80


@pytest.mark.parametrize("parse", PARSERS)
@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (b'{"a": ' * 100000, RecursionError),
        (b"\xff\xfe", UnicodeDecodeError),
        (b'{"a": 1,}', ValueError),
    ],
)
def test_other_malformed_json_keeps_its_route(
    parse: object, raw: bytes, error: type[Exception]
) -> None:
    assert callable(parse)
    with pytest.raises(error):
        parse(raw)
