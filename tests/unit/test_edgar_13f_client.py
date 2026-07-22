import math
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from quantpulse.ingestion import edgar_13f_client


def _fake_settings(tmp_path: Path) -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    return settings


def _write_synthetic_zip(path: Path, *, infotable_tsv: str, submission_tsv: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("INFOTABLE.tsv", infotable_tsv)
        archive.writestr("SUBMISSION.tsv", submission_tsv)


_INFOTABLE_HEADER = (
    "ACCESSION_NUMBER\tINFOTABLE_SK\tNAMEOFISSUER\tTITLEOFCLASS\tCUSIP\tFIGI\tVALUE\t"
    "SSHPRNAMT\tSSHPRNAMTTYPE\tPUTCALL\tINVESTMENTDISCRETION\tOTHERMANAGER\t"
    "VOTING_AUTH_SOLE\tVOTING_AUTH_SHARED\tVOTING_AUTH_NONE"
)
_SUBMISSION_HEADER = "ACCESSION_NUMBER\tFILING_DATE\tSUBMISSIONTYPE\tCIK\tPERIODOFREPORT"


def _infotable_row(
    accession: str, sk: int, issuer: str, value: int, shares: int, *, cusip: str = "000000000"
) -> str:
    return "\t".join(
        [
            accession,
            str(sk),
            issuer,
            "COM",
            cusip,
            "",
            str(value),
            str(shares),
            "SH",
            "",
            "SOLE",
            "",
            str(shares),
            "0",
            "0",
        ]
    )


def _submission_row(
    accession: str, filing_date: str, submission_type: str, cik: str, period: str
) -> str:
    return "\t".join([accession, filing_date, submission_type, cik, period])


# --- quarter_window_for / _prior_quarter_window --------------------------------
# Expected values verified directly against SEC's real, currently published
# bulk-file URL list (module docstring) -- not derived from the code itself.


@pytest.mark.parametrize(
    ("as_of", "expected_start", "expected_end"),
    [
        (date(2026, 7, 22), date(2026, 6, 1), date(2026, 8, 31)),
        (date(2026, 2, 15), date(2025, 12, 1), date(2026, 2, 28)),
        (date(2026, 4, 1), date(2026, 3, 1), date(2026, 5, 31)),
        (date(2025, 10, 10), date(2025, 9, 1), date(2025, 11, 30)),
        (date(2025, 7, 1), date(2025, 6, 1), date(2025, 8, 31)),
        (date(2024, 12, 31), date(2024, 12, 1), date(2025, 2, 28)),
    ],
)
def test_quarter_window_for_matches_real_sec_windows(
    as_of: date, expected_start: date, expected_end: date
) -> None:
    assert edgar_13f_client.quarter_window_for(as_of) == (expected_start, expected_end)


def test_prior_quarter_window_chains_correctly() -> None:
    window = edgar_13f_client.quarter_window_for(date(2026, 4, 1))
    prior = edgar_13f_client._prior_quarter_window(window)
    assert prior == edgar_13f_client.quarter_window_for(date(2026, 1, 1))
    assert prior == (date(2025, 12, 1), date(2026, 2, 28))


def test_bulk_zip_url_matches_real_observed_pattern() -> None:
    window = (date(2026, 3, 1), date(2026, 5, 31))
    url = edgar_13f_client._bulk_zip_url(window)
    assert url == (
        "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
        "01mar2026-31may2026_form13f.zip"
    )


# --- _normalize_issuer_name -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("EBAY INC.", "EBAY"),
        ("APPLE INC", "APPLE"),
        ("MICROSOFT CORP", "MICROSOFT"),
        ("BERKSHIRE HATHAWAY INC", "BERKSHIRE HATHAWAY"),
        ("3M CO", "3M"),
    ],
)
def test_normalize_issuer_name(raw_name: str, expected: str) -> None:
    assert edgar_13f_client._normalize_issuer_name(raw_name) == expected


def test_normalize_issuer_name_matches_ticker_universe_style_names() -> None:
    # Both sides of the join must normalize to the same key.
    assert edgar_13f_client._normalize_issuer_name(
        "Apple Inc."
    ) == edgar_13f_client._normalize_issuer_name("APPLE INC")


# --- ensure_bulk_zip_downloaded --------------------------------------------------


def test_ensure_bulk_zip_downloaded_fetches_and_caches(tmp_path: Path) -> None:
    window = (date(2026, 3, 1), date(2026, 5, 31))
    with (
        patch(
            "quantpulse.ingestion.edgar_13f_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch(
            "quantpulse.ingestion.edgar_13f_client.get_bytes", return_value=b"PK\x03\x04fakezip"
        ) as mock_get_bytes,
    ):
        path = edgar_13f_client.ensure_bulk_zip_downloaded(window)

    assert path.exists()
    assert path.read_bytes() == b"PK\x03\x04fakezip"
    mock_get_bytes.assert_called_once()
    _, kwargs = mock_get_bytes.call_args
    assert kwargs["timeout"] == edgar_13f_client._DOWNLOAD_TIMEOUT_SECONDS


def test_ensure_bulk_zip_downloaded_reuses_existing_file(tmp_path: Path) -> None:
    window = (date(2026, 3, 1), date(2026, 5, 31))
    with patch(
        "quantpulse.ingestion.edgar_13f_client.get_settings", return_value=_fake_settings(tmp_path)
    ):
        path = edgar_13f_client._local_zip_path(window)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"already-here")

        with patch("quantpulse.ingestion.edgar_13f_client.get_bytes") as mock_get_bytes:
            result = edgar_13f_client.ensure_bulk_zip_downloaded(window)

    assert result == path
    mock_get_bytes.assert_not_called()


# --- fetch_quarterly_institutional_holdings -------------------------------------


@pytest.fixture
def universe() -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": ["AAPL", "MSFT"], "name": ["Apple Inc.", "Microsoft Corporation"]}
    )


def test_fetch_quarterly_institutional_holdings_aggregates_and_filters_stale_period(
    tmp_path: Path, universe: pd.DataFrame
) -> None:
    infotable = "\n".join(
        [
            _INFOTABLE_HEADER,
            # Two filers holding AAPL this quarter, one holding MSFT.
            _infotable_row("0001-26-000001", 1, "APPLE INC", 1000000, 5000, cusip="037833100"),
            _infotable_row("0001-26-000001", 2, "MICROSOFT CORP", 2000000, 3000, cusip="594918104"),
            _infotable_row("0002-26-000001", 1, "APPLE INC", 500000, 2500, cusip="037833100"),
            _infotable_row("0002-26-000001", 2, "RANDOM UNRELATED CO", 100, 10, cusip="999999999"),
            # A late amendment referencing an OLDER period -- must be excluded.
            _infotable_row("0003-26-000001", 1, "APPLE INC", 250000, 9999, cusip="037833100"),
        ]
    )
    submission = "\n".join(
        [
            _SUBMISSION_HEADER,
            _submission_row("0001-26-000001", "15-APR-2026", "13F-HR", "0001111111", "31-MAR-2026"),
            _submission_row("0002-26-000001", "20-APR-2026", "13F-HR", "0002222222", "31-MAR-2026"),
            _submission_row(
                "0003-26-000001", "25-APR-2026", "13F-NT/A", "0003333333", "31-DEC-2025"
            ),
        ]
    )
    zip_path = tmp_path / "synthetic.zip"
    _write_synthetic_zip(zip_path, infotable_tsv=infotable, submission_tsv=submission)

    window = edgar_13f_client.quarter_window_for(date(2026, 4, 1))
    with patch.object(edgar_13f_client, "ensure_bulk_zip_downloaded", return_value=zip_path):
        result = edgar_13f_client.fetch_quarterly_institutional_holdings(window, universe)

    assert list(result.columns) == edgar_13f_client._HOLDINGS_COLUMNS
    aapl = result[result["symbol"] == "AAPL"].iloc[0]
    msft = result[result["symbol"] == "MSFT"].iloc[0]

    # The stale Dec-2025 row (9999 shares) must NOT be included in AAPL's total.
    assert aapl["total_shares_held"] == 7500
    assert aapl["total_value"] == 1500000
    assert aapl["num_filers"] == 2
    assert str(aapl["quarter_end_date"]) == "2026-03-31"

    assert msft["total_shares_held"] == 3000
    assert msft["total_value"] == 2000000
    assert msft["num_filers"] == 1

    # The unrelated issuer must not leak into results at all.
    assert "RANDOM UNRELATED CO" not in result.to_string()


def test_fetch_quarterly_institutional_holdings_empty_universe_skips_download(
    tmp_path: Path,
) -> None:
    empty_universe = pd.DataFrame({"symbol": [], "name": []})
    window = edgar_13f_client.quarter_window_for(date(2026, 4, 1))
    with patch.object(edgar_13f_client, "ensure_bulk_zip_downloaded") as mock_ensure:
        result = edgar_13f_client.fetch_quarterly_institutional_holdings(window, empty_universe)
    assert result.empty
    assert list(result.columns) == edgar_13f_client._HOLDINGS_COLUMNS
    # The whole point of the fix: an unmatchable universe must not trigger
    # the ~100MB bulk download at all.
    mock_ensure.assert_not_called()


def test_fetch_quarterly_institutional_holdings_no_matches_returns_empty_shape(
    tmp_path: Path, universe: pd.DataFrame
) -> None:
    infotable = "\n".join(
        [_INFOTABLE_HEADER, _infotable_row("0001-26-000001", 1, "SOME OTHER COMPANY", 100, 10)]
    )
    submission = "\n".join(
        [
            _SUBMISSION_HEADER,
            _submission_row("0001-26-000001", "15-APR-2026", "13F-HR", "0001111111", "31-MAR-2026"),
        ]
    )
    zip_path = tmp_path / "synthetic.zip"
    _write_synthetic_zip(zip_path, infotable_tsv=infotable, submission_tsv=submission)

    window = edgar_13f_client.quarter_window_for(date(2026, 4, 1))
    with patch.object(edgar_13f_client, "ensure_bulk_zip_downloaded", return_value=zip_path):
        result = edgar_13f_client.fetch_quarterly_institutional_holdings(window, universe)
    assert result.empty
    assert list(result.columns) == edgar_13f_client._HOLDINGS_COLUMNS


# --- fetch_institutional_ownership_trend ----------------------------------------


def test_fetch_institutional_ownership_trend_computes_change_and_honest_nan(
    tmp_path: Path, universe: pd.DataFrame
) -> None:
    window = edgar_13f_client.quarter_window_for(date(2026, 4, 1))
    prior_window = edgar_13f_client._prior_quarter_window(window)

    current_infotable = "\n".join(
        [
            _INFOTABLE_HEADER,
            _infotable_row("0001-26-000001", 1, "APPLE INC", 1000000, 5000, cusip="037833100"),
            _infotable_row("0002-26-000001", 1, "APPLE INC", 500000, 2500, cusip="037833100"),
            _infotable_row("0003-26-000001", 1, "MICROSOFT CORP", 2000000, 3000, cusip="594918104"),
        ]
    )
    current_submission = "\n".join(
        [
            _SUBMISSION_HEADER,
            _submission_row("0001-26-000001", "15-APR-2026", "13F-HR", "0001111111", "31-MAR-2026"),
            _submission_row("0002-26-000001", "20-APR-2026", "13F-HR", "0002222222", "31-MAR-2026"),
            _submission_row("0003-26-000001", "20-APR-2026", "13F-HR", "0002222222", "31-MAR-2026"),
        ]
    )
    current_zip = tmp_path / "current.zip"
    _write_synthetic_zip(
        current_zip, infotable_tsv=current_infotable, submission_tsv=current_submission
    )

    # Prior quarter: AAPL held (smaller position), MSFT not held by anyone.
    prior_infotable = "\n".join(
        [
            _INFOTABLE_HEADER,
            _infotable_row("0010-26-000001", 1, "APPLE INC", 900000, 4000, cusip="037833100"),
            _infotable_row("0011-26-000001", 1, "APPLE INC", 400000, 2000, cusip="037833100"),
        ]
    )
    prior_submission = "\n".join(
        [
            _SUBMISSION_HEADER,
            _submission_row("0010-26-000001", "15-JAN-2026", "13F-HR", "0001111111", "31-DEC-2025"),
            _submission_row("0011-26-000001", "20-JAN-2026", "13F-HR", "0002222222", "31-DEC-2025"),
        ]
    )
    prior_zip = tmp_path / "prior.zip"
    _write_synthetic_zip(prior_zip, infotable_tsv=prior_infotable, submission_tsv=prior_submission)

    paths_by_window = {window: current_zip, prior_window: prior_zip}

    def _fake_ensure(w: tuple[date, date]) -> Path:
        return paths_by_window[w]

    with patch.object(edgar_13f_client, "ensure_bulk_zip_downloaded", side_effect=_fake_ensure):
        result = edgar_13f_client.fetch_institutional_ownership_trend(window, universe)

    assert list(result.columns) == edgar_13f_client._TREND_COLUMNS
    aapl = result[result["symbol"] == "AAPL"].iloc[0]
    msft = result[result["symbol"] == "MSFT"].iloc[0]

    # AAPL: current 7500 shares, prior 4000+2000=6000 -> +1500
    assert aapl["change_from_prior_quarter"] == 1500

    # MSFT: no prior match at all -> honest NaN, never a filled 0.
    assert math.isnan(msft["change_from_prior_quarter"])


def test_fetch_institutional_ownership_trend_empty_current_returns_empty_shape(
    tmp_path: Path, universe: pd.DataFrame
) -> None:
    window = edgar_13f_client.quarter_window_for(date(2026, 4, 1))
    empty_infotable = _INFOTABLE_HEADER
    empty_submission = _SUBMISSION_HEADER
    empty_zip = tmp_path / "empty.zip"
    _write_synthetic_zip(empty_zip, infotable_tsv=empty_infotable, submission_tsv=empty_submission)

    with patch.object(edgar_13f_client, "ensure_bulk_zip_downloaded", return_value=empty_zip):
        result = edgar_13f_client.fetch_institutional_ownership_trend(window, universe)

    assert result.empty
    assert list(result.columns) == edgar_13f_client._TREND_COLUMNS
