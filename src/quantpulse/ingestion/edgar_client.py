import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_dataframe, cached_json
from quantpulse.ingestion.circuit_breaker import get_breaker
from quantpulse.ingestion.http import get_json, get_text
from quantpulse.ingestion.rate_limit import SimpleRateLimiter

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
_SOURCE = "edgar"

# Form 4 = original filing, Form 4/A = amendment. Forms 3 (initial ownership)
# and 5 (annual) exist too but aren't transaction reports, so they're outside
# this signal's scope (Section 24: insider *transactions*).
_FORM4_FORMS = frozenset({"4", "4/A"})

# Section 5: "free, no key, generous fair-use rate". No documented number,
# but SEC's own guidance is to stay well under ~10 req/sec -- min-interval,
# not a burst-allowing token bucket, is the polite fit for a fair-use source.
_rate_limiter = SimpleRateLimiter(min_interval_seconds=0.2)


def _cache_dir(subdir: str) -> Path:
    return Path(get_settings().ingestion_cache_dir) / "edgar" / subdir


def _headers() -> dict[str, str]:
    return {"User-Agent": get_settings().sec_edgar_user_agent}


def fetch_cik_lookup() -> pd.DataFrame:
    """Ticker -> CIK mapping (SEC's own file, not company-facts specific)."""

    def _fetch() -> pd.DataFrame:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            result = get_json(_TICKERS_URL, headers=_headers())
        df = pd.DataFrame(result.values())
        return df.rename(columns={"cik_str": "cik", "title": "name"})[["ticker", "cik", "name"]]

    return cached_dataframe("cik_lookup", _fetch, _cache_dir("cik_lookup"), ttl=timedelta(days=30))


def get_cik_for_ticker(symbol: str) -> str:
    """10-digit, zero-padded CIK for `symbol`, as required by the company-facts URL."""
    lookup = fetch_cik_lookup()
    matches = lookup[lookup["ticker"].str.upper() == symbol.upper()]
    if matches.empty:
        raise ValueError(f"No CIK found for ticker {symbol!r}")
    return f"{int(matches.iloc[0]['cik']):010d}"


def fetch_company_facts(symbol: str) -> dict[str, Any]:
    """Raw XBRL company-facts payload (all reported financial-statement concepts) for `symbol`."""
    cik = get_cik_for_ticker(symbol)

    def _fetch() -> dict[str, Any]:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            return dict(get_json(_COMPANY_FACTS_URL.format(cik=cik), headers=_headers()))

    return dict(
        cached_json(
            f"company_facts_{symbol}", _fetch, _cache_dir("company_facts"), ttl=timedelta(days=7)
        )
    )


def fetch_recent_filings(
    symbol: str, *, forms: frozenset[str], lookback_days: int
) -> list[dict[str, Any]]:
    """Recent filings of the given `forms` for `symbol`, from its SEC submissions index.

    Returns `[{"accession_number", "filing_date", "primary_document"}, ...]`,
    newest first isn't guaranteed (submissions.json's own order is used) --
    callers that care about order should sort explicitly.
    """
    cik = get_cik_for_ticker(symbol)

    def _fetch() -> dict[str, Any]:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            return dict(get_json(_SUBMISSIONS_URL.format(cik=cik), headers=_headers()))

    # New filings can appear daily; a half-day TTL balances freshness against
    # not re-fetching the whole (often large) submissions history every call.
    submissions = cached_json(
        f"submissions_{symbol}", _fetch, _cache_dir("submissions"), ttl=timedelta(hours=12)
    )
    recent = submissions.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    if not forms_list:
        return []

    cutoff = date.today() - timedelta(days=lookback_days)
    filings = []
    for i, form in enumerate(forms_list):
        if form not in forms:
            continue
        try:
            filing_date = date.fromisoformat(recent["filingDate"][i])
        except (ValueError, IndexError):
            continue
        if filing_date < cutoff:
            continue
        filings.append(
            {
                "accession_number": recent["accessionNumber"][i],
                "filing_date": filing_date,
                "primary_document": recent["primaryDocument"][i],
            }
        )
    return filings


def _filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Raw XML data-document URL for a filing.

    `primary_document` from submissions.json (e.g. "xslF345X06/form4.xml")
    is the path to SEC's auto-generated HTML *viewer* for structured-XML
    forms -- the raw XML a program should parse sits at the same basename in
    the accession folder's root, not nested under that "xslXXX/" viewer
    subfolder. Verified directly against a real filing (fetching the
    `primaryDocument` path returns an HTML wrapper, not the ownership XML;
    the accession folder's own index lists the raw file at top level under
    the identical basename), not assumed.
    """
    accession_no_dashes = accession_number.replace("-", "")
    cik_no_zeros = str(int(cik))
    basename = primary_document.rsplit("/", 1)[-1]
    return _ARCHIVE_URL.format(cik=cik_no_zeros, accession=accession_no_dashes, document=basename)


def _xml_float(element: ET.Element, path: str) -> float | None:
    """`element.findtext(path)` parsed as a float, or None if absent/non-numeric.

    Several Form-4 numeric fields (notably `transactionPricePerShare` on an
    option exercise) are legitimately reported as only a footnote reference
    with no `<value>` child at all -- verified against a real filing, not a
    hypothetical -- so a missing value here means "not reported," not a bug.
    """
    raw = element.findtext(path)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_transaction(element: ET.Element, *, security_kind: str) -> dict[str, Any]:
    return {
        "security_kind": security_kind,
        "transaction_date": element.findtext("transactionDate/value"),
        "transaction_code": element.findtext("transactionCoding/transactionCode"),
        "shares": _xml_float(element, "transactionAmounts/transactionShares/value"),
        "price_per_share": _xml_float(element, "transactionAmounts/transactionPricePerShare/value"),
        "acquired_disposed_code": element.findtext(
            "transactionAmounts/transactionAcquiredDisposedCode/value"
        ),
        "shares_owned_after": _xml_float(
            element, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"
        ),
    }


def _parse_form4_xml(raw_xml: str, *, symbol: str) -> list[dict[str, Any]]:
    """One Form-4 ownership XML document -> its non-derivative (common-stock) transaction rows.

    Derivative-table transactions (option/RSU exercises, etc.) are a
    distinct, noisier category intentionally left out here -- Section 13's
    `insider_transactions` schema is a flat common-stock view, matching
    Section 24's framing of the signal as insider buy/sell activity.
    """
    root = ET.fromstring(raw_xml)
    issuer_symbol = root.findtext("issuer/issuerTradingSymbol") or symbol
    report_date = root.findtext("periodOfReport")
    insider_name = root.findtext("reportingOwner/reportingOwnerId/rptOwnerName")
    insider_title = root.findtext("reportingOwner/reportingOwnerRelationship/officerTitle")

    rows = []
    for transaction in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        row = _parse_transaction(transaction, security_kind="non_derivative")
        row.update(
            {
                "symbol": issuer_symbol,
                "insider_name": insider_name,
                "insider_title": insider_title,
                "report_date": report_date,
            }
        )
        rows.append(row)
    return rows


def fetch_form4_transactions(
    symbol: str, accession_number: str, primary_document: str
) -> list[dict[str, Any]]:
    """Parsed non-derivative transaction rows from one Form-4 filing's raw ownership XML."""
    cik = get_cik_for_ticker(symbol)
    url = _filing_document_url(cik, accession_number, primary_document)

    def _fetch() -> list[dict[str, Any]]:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            raw_xml = get_text(url, headers=_headers())
        return _parse_form4_xml(raw_xml, symbol=symbol)

    # A filed document is immutable once submitted -- cache indefinitely
    # (long TTL, not literally forever, so a corrupted cache entry can heal).
    return list(
        cached_json(
            f"form4_{accession_number}",
            _fetch,
            _cache_dir("form4_filings"),
            ttl=timedelta(days=365),
        )
    )


_INSIDER_TRANSACTION_COLUMNS = [
    "symbol",
    "insider_name",
    "insider_title",
    "filing_date",
    "report_date",
    "security_kind",
    "transaction_date",
    "transaction_code",
    "acquired_disposed_code",
    "shares",
    "price_per_share",
    "shares_owned_after",
]


def fetch_insider_transactions(
    symbol: str, *, lookback_days: int = 180, max_filings: int = 40
) -> pd.DataFrame:
    """Form-4 insider buy/sell transactions for `symbol` over the trailing `lookback_days`.

    Section 24.

    Lists recent Form 4 / 4-A filings from the company's SEC submissions
    index, then fetches and parses each filing's raw ownership XML.
    `max_filings` bounds how many individual filing documents get fetched in
    one call (each is its own HTTP request, most recent first); every filing
    is cached indefinitely once parsed, so repeat calls only pay for
    genuinely new filings.

    Interpreting these rows into a net "insider buying vs. selling" reading
    -- Section 24's real point being that *clusters* of several distinct
    insiders acting together matter far more than any single trade -- is the
    aggregation module's job, not this ingestion client's: this returns raw,
    per-transaction rows faithfully, including `price_per_share=None` for
    transactions (e.g. option exercises) SEC's own filing reports only via a
    footnote rather than a number.
    """
    filings = fetch_recent_filings(symbol, forms=_FORM4_FORMS, lookback_days=lookback_days)
    filings = sorted(filings, key=lambda f: f["filing_date"], reverse=True)[:max_filings]

    rows: list[dict[str, Any]] = []
    for filing in filings:
        transactions = fetch_form4_transactions(
            symbol, filing["accession_number"], filing["primary_document"]
        )
        for transaction in transactions:
            rows.append({**transaction, "filing_date": filing["filing_date"]})

    df = pd.DataFrame(rows, columns=_INSIDER_TRANSACTION_COLUMNS)
    if not df.empty:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.date
        df["report_date"] = pd.to_datetime(df["report_date"]).dt.date
    return df
