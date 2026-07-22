from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from quantpulse.ingestion import edgar_client

# Shaped after a real filing fetched and inspected directly against SEC EDGAR
# during development (not a guess): a non-derivative table with one
# option-exercise ("M", price only in a footnote -- no <value>) and one
# tax-withholding disposition ("F", a real numeric price).
_SAMPLE_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0609</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2026-06-15</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001780525</rptOwnerCik>
            <rptOwnerName>Newstead Jennifer</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isOfficer>true</isOfficer>
            <officerTitle>SVP, GC and Secretary</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2026-06-15</value></transactionDate>
            <transactionCoding>
                <transactionCode>M</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>30104</value></transactionShares>
                <transactionPricePerShare><footnoteId id="F1"/></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>57784</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <transactionDate><value>2026-06-15</value></transactionDate>
            <transactionCoding>
                <transactionCode>F</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>16238</value></transactionShares>
                <transactionPricePerShare><value>296.42</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>41546</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <derivativeTable>
        <derivativeTransaction>
            <transactionDate><value>2026-06-15</value></transactionDate>
            <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
        </derivativeTransaction>
    </derivativeTable>
</ownershipDocument>
"""


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(edgar_client._rate_limiter, "wait", lambda: None)


def _fake_settings(tmp_path: Path) -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    settings.sec_edgar_user_agent = "test-agent test@example.com"
    return settings


def test_get_cik_for_ticker_looks_up_and_pads(tmp_path: Path) -> None:
    lookup_response = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    }
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.edgar_client.get_json", return_value=lookup_response),
    ):
        cik = edgar_client.get_cik_for_ticker("aapl")

    assert cik == "0000320193"


def test_get_cik_for_ticker_raises_for_unknown_ticker(tmp_path: Path) -> None:
    lookup_response = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.edgar_client.get_json", return_value=lookup_response),
    ):
        with pytest.raises(ValueError):
            edgar_client.get_cik_for_ticker("NOPE")


def test_fetch_company_facts_uses_padded_cik_in_url(tmp_path: Path) -> None:
    lookup_response = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    facts_response = {"cik": 320193, "entityName": "Apple Inc.", "facts": {}}

    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.edgar_client.get_json",
            side_effect=[lookup_response, facts_response],
        ) as mock_get_json,
    ):
        result = edgar_client.fetch_company_facts("AAPL")

    assert result["entityName"] == "Apple Inc."
    called_url = mock_get_json.call_args_list[1].args[0]
    assert "CIK0000320193.json" in called_url


# --- fetch_recent_filings -----------------------------------------------------


def _submissions_payload(forms_and_dates: list[tuple[str, str]]) -> dict:
    return {
        "filings": {
            "recent": {
                "form": [f for f, _ in forms_and_dates],
                "filingDate": [d for _, d in forms_and_dates],
                "accessionNumber": [f"0001-26-{i:06d}" for i in range(len(forms_and_dates))],
                "primaryDocument": [
                    f"xslF345X06/form4_{i}.xml" for i in range(len(forms_and_dates))
                ],
            }
        }
    }


def _cik_lookup_response() -> dict:
    return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}


def test_fetch_recent_filings_filters_by_form_and_lookback(tmp_path: Path) -> None:
    today = date(2026, 7, 22)
    payload = _submissions_payload(
        [
            ("4", "2026-07-20"),  # form 4, within lookback -- kept
            ("10-K", "2026-07-20"),  # wrong form -- dropped
            ("4/A", "2026-01-01"),  # form 4/A, too old -- dropped
            ("4", "2026-06-01"),  # form 4, within lookback -- kept
        ]
    )
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.edgar_client.get_json",
            side_effect=[_cik_lookup_response(), payload],
        ),
        patch("quantpulse.ingestion.edgar_client.date") as mock_date,
    ):
        mock_date.today.return_value = today
        mock_date.fromisoformat = date.fromisoformat
        filings = edgar_client.fetch_recent_filings(
            "AAPL", forms=edgar_client._FORM4_FORMS, lookback_days=90
        )

    assert len(filings) == 2
    assert {f["accession_number"] for f in filings} == {"0001-26-000000", "0001-26-000003"}


def test_fetch_recent_filings_empty_when_no_recent_filings(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.edgar_client.get_json",
            side_effect=[_cik_lookup_response(), {"filings": {"recent": {}}}],
        ),
    ):
        filings = edgar_client.fetch_recent_filings(
            "AAPL", forms=edgar_client._FORM4_FORMS, lookback_days=90
        )
    assert filings == []


# --- _filing_document_url -----------------------------------------------------


def test_filing_document_url_strips_xsl_viewer_prefix() -> None:
    url = edgar_client._filing_document_url(
        "0000320193", "0001140361-26-025622", "xslF345X06/form4.xml"
    )
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000114036126025622/form4.xml"


def test_filing_document_url_handles_document_with_no_viewer_prefix() -> None:
    url = edgar_client._filing_document_url("0000320193", "0001140361-26-025622", "form4.xml")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000114036126025622/form4.xml"


# --- XML parsing ---------------------------------------------------------------


def test_parse_form4_xml_extracts_non_derivative_transactions_only() -> None:
    rows = edgar_client._parse_form4_xml(_SAMPLE_FORM4_XML, symbol="AAPL")

    assert len(rows) == 2  # the single derivativeTransaction is excluded
    assert all(r["security_kind"] == "non_derivative" for r in rows)


def test_parse_form4_xml_uses_issuer_symbol_from_document() -> None:
    rows = edgar_client._parse_form4_xml(_SAMPLE_FORM4_XML, symbol="WRONG")
    assert all(r["symbol"] == "AAPL" for r in rows)  # issuer XML wins over the passed-in symbol


def test_parse_form4_xml_extracts_insider_identity() -> None:
    rows = edgar_client._parse_form4_xml(_SAMPLE_FORM4_XML, symbol="AAPL")
    assert rows[0]["insider_name"] == "Newstead Jennifer"
    assert rows[0]["insider_title"] == "SVP, GC and Secretary"
    assert rows[0]["report_date"] == "2026-06-15"


def test_parse_form4_xml_footnoted_price_is_none_not_a_crash() -> None:
    # Regression: an option-exercise ("M") transaction's price is reported
    # only via a footnote, with no <value> child at all.
    rows = edgar_client._parse_form4_xml(_SAMPLE_FORM4_XML, symbol="AAPL")
    exercise_row = next(r for r in rows if r["transaction_code"] == "M")
    assert exercise_row["price_per_share"] is None
    assert exercise_row["shares"] == 30104.0
    assert exercise_row["acquired_disposed_code"] == "A"


def test_parse_form4_xml_normal_transaction_has_numeric_price() -> None:
    rows = edgar_client._parse_form4_xml(_SAMPLE_FORM4_XML, symbol="AAPL")
    withholding_row = next(r for r in rows if r["transaction_code"] == "F")
    assert withholding_row["price_per_share"] == 296.42
    assert withholding_row["shares"] == 16238.0
    assert withholding_row["shares_owned_after"] == 41546.0
    assert withholding_row["acquired_disposed_code"] == "D"


# --- fetch_form4_transactions / fetch_insider_transactions --------------------


def test_fetch_form4_transactions_fetches_and_parses(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.edgar_client.get_json", return_value=_cik_lookup_response()),
        patch(
            "quantpulse.ingestion.edgar_client.get_text", return_value=_SAMPLE_FORM4_XML
        ) as mock_get_text,
    ):
        rows = edgar_client.fetch_form4_transactions("AAPL", "0001140361-26-025622", "form4.xml")

    assert len(rows) == 2
    called_url = mock_get_text.call_args.args[0]
    assert "000114036126025622" in called_url


def test_fetch_insider_transactions_assembles_dataframe(tmp_path: Path) -> None:
    today = date(2026, 7, 22)
    payload = _submissions_payload([("4", "2026-07-20")])
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.edgar_client.get_json",
            side_effect=[_cik_lookup_response(), payload, _cik_lookup_response()],
        ),
        patch("quantpulse.ingestion.edgar_client.get_text", return_value=_SAMPLE_FORM4_XML),
        patch("quantpulse.ingestion.edgar_client.date") as mock_date,
    ):
        mock_date.today.return_value = today
        mock_date.fromisoformat = date.fromisoformat
        df = edgar_client.fetch_insider_transactions("AAPL", lookback_days=90, max_filings=10)

    assert list(df.columns) == edgar_client._INSIDER_TRANSACTION_COLUMNS
    assert len(df) == 2
    assert (df["symbol"] == "AAPL").all()
    assert df["filing_date"].iloc[0] == date(2026, 7, 20)
    assert df["transaction_date"].iloc[0] == date(2026, 6, 15)


def test_fetch_insider_transactions_empty_when_no_filings(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.edgar_client.get_json",
            side_effect=[_cik_lookup_response(), {"filings": {"recent": {}}}],
        ),
    ):
        df = edgar_client.fetch_insider_transactions("AAPL")

    assert df.empty
    assert list(df.columns) == edgar_client._INSIDER_TRANSACTION_COLUMNS


def test_fetch_insider_transactions_respects_max_filings(tmp_path: Path) -> None:
    today = date(2026, 7, 22)
    payload = _submissions_payload([("4", "2026-07-20"), ("4", "2026-07-15"), ("4", "2026-07-10")])
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.edgar_client.get_json",
            side_effect=[_cik_lookup_response(), payload] + [_cik_lookup_response()] * 3,
        ),
        patch("quantpulse.ingestion.edgar_client.get_text", return_value=_SAMPLE_FORM4_XML),
        patch("quantpulse.ingestion.edgar_client.date") as mock_date,
    ):
        mock_date.today.return_value = today
        mock_date.fromisoformat = date.fromisoformat
        df = edgar_client.fetch_insider_transactions("AAPL", lookback_days=90, max_filings=1)

    # Only the single most recent filing's 2 transactions, not all 3 filings'.
    assert len(df) == 2


def test_fetch_recent_filings_lookback_boundary_is_real_timedelta(tmp_path: Path) -> None:
    # Sanity check that filtering genuinely uses lookback_days, not a stub.
    today = date(2026, 7, 22)
    payload = _submissions_payload([("4", (today - timedelta(days=5)).isoformat())])
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.edgar_client.get_json",
            side_effect=[_cik_lookup_response(), payload],
        ),
        patch("quantpulse.ingestion.edgar_client.date") as mock_date,
    ):
        mock_date.today.return_value = today
        mock_date.fromisoformat = date.fromisoformat
        filings = edgar_client.fetch_recent_filings(
            "AAPL", forms=edgar_client._FORM4_FORMS, lookback_days=3
        )
    assert filings == []
