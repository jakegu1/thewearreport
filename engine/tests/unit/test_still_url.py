"""The still URL a camera record's URL field gives (T-066): a string, or an object's
string member "url"."""

from __future__ import annotations

import pytest

from wearreport.tools import pilot_heights as ph


@pytest.mark.parametrize(
    ("value", "url"),
    [
        ("http://h/a.jpg", "http://h/a.jpg"),
        ("", ""),
        ({"url": "http://h/a.jpg", "description": "Camera 1"}, "http://h/a.jpg"),
        ({"url": "http://h/a.jpg"}, "http://h/a.jpg"),
        ({"description": "Camera 1"}, None),
        ({"url": None}, None),
        ({"url": 1}, None),
        ({"url": {"url": "http://h/a.jpg"}}, None),
        ({"url": ["http://h/a.jpg"]}, None),
        (["http://h/a.jpg"], None),
        (1, None),
        (None, None),
        (True, None),
    ],
)
def test_still_url(value: object, url: str | None) -> None:
    assert ph._still_url(value) == url


def test_the_object_form_is_read_for_any_named_field() -> None:
    point = {"type": "Point", "coordinates": [-97.745, 30.27]}
    record = {"screenshot_address": {"url": "https://x/1.jpg"}, "location": point}
    assert ph._camera(record, ("screenshot_address", "location")) == (
        30.27,
        -97.745,
        "https://x/1.jpg",
    )
