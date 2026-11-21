"""Pure helpers, no database needed."""

import datetime as dt
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.fmt import epoch_ms, iso  # noqa: E402
from app.queries import BadRequest, enum_param, is_h3_cell, rating_value, spearman  # noqa: E402

UTC = dt.timezone.utc


def test_iso_format_is_the_contract():
    t = dt.datetime(2026, 11, 21, 9, 12, 3, 512345, tzinfo=UTC)
    assert iso(t) == "2026-11-21T09:12:03.512Z"
    assert iso(t.astimezone(dt.timezone(dt.timedelta(hours=2)))) == "2026-11-21T09:12:03.512Z"
    assert iso(None) is None
    assert epoch_ms(t) == 1795252323512 and isinstance(epoch_ms(t), int)


def test_enum_param():
    assert enum_param(None, "x", ("a",)) is None and enum_param("", "x", ("a",)) is None
    assert enum_param("a", "x", ("a",)) == "a"
    try:
        enum_param("b", "x", ("a",))
        raise AssertionError("expected BadRequest")
    except BadRequest as e:
        assert "x must be one of a" in str(e)


def test_h3_cell():
    assert is_h3_cell("8a1fb46622dffff")
    assert not is_h3_cell("8a1fb46622dfff") and not is_h3_cell("7a1fb46622dffff")
    assert not is_h3_cell("8a1fb46622dfffz")


def test_rating_value():
    assert rating_value("3") == 3 and rating_value("4.0") == 4 and rating_value(5) == 5
    assert rating_value(None) is None and rating_value("") is None and rating_value("skip") is None
    assert rating_value("inf") is None and rating_value("nan") is None


def test_telemetry_off_without_connection_string(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("psycopg")
    from app.main import configure_telemetry

    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    assert configure_telemetry() is False


def test_spearman_ranks_with_ties_like_the_mock():
    assert spearman([0.1, 0.5, 0.9], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([0.1, 0.5, 0.9], [3, 2, 1]) == pytest.approx(-1.0)
    # Average ranks for ties: x ranks [0, 1.5, 1.5, 3], y ranks [0, 1, 2, 3].
    assert spearman([1.0, 2.0, 2.0, 3.0], [0, 1, 2, 3]) == pytest.approx(4.5 / (4.5 * 5.0) ** 0.5)
    assert spearman([1.0, 2.0], [1, 2]) is None  # fewer than 3 ratings
    assert spearman([1.0, 2.0, 3.0], [4, 4, 4]) is None  # constant ratings: undefined
