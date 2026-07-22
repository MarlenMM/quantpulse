from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from quantpulse.ingestion import historical_constituents_client as hist
from quantpulse.ingestion.historical_constituents_client import HistoricalMembershipUnavailable


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "BRK.B", "AABA", "BADROW"],
            "start_date": ["1996-01-02", "2010-02-16", "1999-12-08", "not-a-date"],
            "end_date": [None, None, "2017-06-19", None],
        }
    )


def test_parse_interval_frame_normalizes_and_types() -> None:
    df = hist._parse_interval_frame(_raw())

    # BADROW dropped (unparseable start_date), BRK.B normalized to BRK-B.
    assert set(df["symbol"]) == {"AAPL", "BRK-B", "AABA"}
    aapl = df[df["symbol"] == "AAPL"].iloc[0]
    assert aapl["added_date"] == date(1996, 1, 2)
    assert aapl["removed_date"] is None
    aaba = df[df["symbol"] == "AABA"].iloc[0]
    assert aaba["removed_date"] == date(2017, 6, 19)


def test_parse_interval_frame_rejects_wrong_schema() -> None:
    with pytest.raises(HistoricalMembershipUnavailable):
        hist._parse_interval_frame(pd.DataFrame({"symbol": ["AAPL"], "date": ["2020-01-01"]}))


def test_parse_treats_future_end_date_as_still_open() -> None:
    # Datasets mark a current member's open interval either as empty end_date or
    # a far-future sentinel; both must resolve to None, or a live name would be
    # flagged removed and dropped from a survivorship-aware backtest.
    raw = pd.DataFrame(
        {
            "ticker": ["EMPTY", "SENTINEL", "REALGONE"],
            "start_date": ["2000-01-01", "2001-01-01", "2002-01-01"],
            "end_date": [None, "2059-12-31", "2017-06-19"],
        }
    )
    df = hist._parse_interval_frame(raw, today=date(2026, 7, 22)).set_index("symbol")
    assert df.loc["EMPTY", "removed_date"] is None
    assert df.loc["SENTINEL", "removed_date"] is None  # future -> not yet removed
    assert df.loc["REALGONE", "removed_date"] == date(2017, 6, 19)  # real removal kept


def _settings(**kw: object) -> Mock:
    settings = Mock()
    settings.historical_constituents_path = kw.get("path")
    settings.historical_constituents_url = kw.get("url", "")
    settings.ingestion_cache_dir = str(kw.get("cache_dir", "/tmp"))
    return settings


def test_fetch_reads_local_path_when_set(tmp_path: Path) -> None:
    csv = tmp_path / "hist.csv"
    _raw().to_csv(csv, index=False)

    with patch.object(hist, "get_settings", return_value=_settings(path=str(csv))):
        df = hist.fetch_historical_membership()

    assert set(df["symbol"]) == {"AAPL", "BRK-B", "AABA"}


def test_fetch_raises_when_local_path_missing(tmp_path: Path) -> None:
    with patch.object(
        hist, "get_settings", return_value=_settings(path=str(tmp_path / "nope.csv"))
    ):
        with pytest.raises(HistoricalMembershipUnavailable):
            hist.fetch_historical_membership()


def test_fetch_raises_when_no_source_configured() -> None:
    with patch.object(hist, "get_settings", return_value=_settings(url="")):
        with pytest.raises(HistoricalMembershipUnavailable):
            hist.fetch_historical_membership()


def test_build_current_only_is_survivorship_biased(caplog: pytest.LogCaptureFixture) -> None:
    current = pd.DataFrame({"symbol": ["AAPL", "MSFT", "NEWCO"]})
    dates = pd.DataFrame(
        {"symbol": ["AAPL", "MSFT"], "added_date": [date(1996, 1, 2), date(1994, 6, 1)]}
    )

    with (
        patch.object(hist.wikipedia_client, "fetch_sp500_constituents", return_value=current),
        patch.object(hist.wikipedia_client, "fetch_sp500_date_added", return_value=dates),
        caplog.at_level("WARNING"),
    ):
        df = hist.build_current_only_membership(fallback_added_date=date(1990, 1, 1))

    assert df["removed_date"].isna().all()  # no removals known -> the whole limitation
    assert df[df["symbol"] == "AAPL"].iloc[0]["added_date"] == date(1996, 1, 2)
    # NEWCO had no parseable add-date -> floor used.
    assert df[df["symbol"] == "NEWCO"].iloc[0]["added_date"] == date(1990, 1, 1)
    assert any("SURVIVORSHIP" in r.message for r in caplog.records)
