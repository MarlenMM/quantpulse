"""Institutional ownership trend — SEC Form 13F quarterly bulk data (Section 24).

SEC publishes ALL Form 13F holdings each quarter as one bulk, structured
TSV-in-a-ZIP dataset (`sec.gov/data-research/sec-markets-data/form-13f-data-sets`)
-- free, no key, ~100MB compressed / ~400MB uncompressed per quarter. This is
a genuinely different shape of "ingestion" from every other client in this
package: one huge periodic file to stream-filter, not a small per-symbol REST
call. Verified directly against a real downloaded quarterly file (not
assumed) before writing this module:

- The bulk-file window naming (`01mar2026-31may2026_form13f.zip`) follows a
  fixed quarterly cadence offset one month from calendar quarters:
  Mar-May, Jun-Aug, Sep-Nov, Dec-Feb. `_quarter_window_for` computes this
  rule; the dataset's earliest (2024) window is irregular and isn't
  reproduced by this function, since only current/recent windows matter here.
- `INFOTABLE.tsv`'s `VALUE` column changed units on **2023-01-03**: reported
  in thousands of dollars before that date, in actual dollars since. This
  module assumes actual dollars (correct for any window it would realistically
  be asked to fetch) and does not attempt to handle pre-2023 windows -- silently
  treating an old thousands-denominated file as dollars would be exactly the
  quiet 1000x-wrong-number bug Section 22 warns about.
- There is no free CUSIP-to-ticker mapping available to this project, so
  holdings are matched to the ticker universe by *normalized issuer name*
  text (`NAMEOFISSUER`, e.g. "EBAY INC.") rather than CUSIP -- a real,
  disclosed limitation (a handful of name variants could go unmatched), not
  a silent gap.
"""

import logging
import re
import zipfile
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quantpulse.config import get_settings
from quantpulse.ingestion.circuit_breaker import get_breaker
from quantpulse.ingestion.http import get_bytes, resource_exists
from quantpulse.ingestion.rate_limit import SimpleRateLimiter

logger = logging.getLogger(__name__)

_SOURCE = "edgar_13f"
_BULK_URL_TEMPLATE = (
    "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{start}-{end}_form13f.zip"
)
_MONTH_ABBREVIATIONS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)
_INFOTABLE_CHUNK_SIZE = 200_000
_DOWNLOAD_TIMEOUT_SECONDS = 180.0  # ~100MB over a plain GET needs far more than get_bytes' default

# How many quarters back `latest_published_window` will look for a published
# file. Four is a year: enough to absorb the normal publication lag (weeks) plus
# a long outage, and short enough that a total SEC failure costs four HEAD
# requests rather than an unbounded crawl. Ingesting a window older than a year
# would be worse than ingesting nothing, since it would be presented as "this
# quarter's institutional trend".
_MAX_WINDOW_LOOKBACK = 4

# Same fair-use posture as edgar_client.py.
_rate_limiter = SimpleRateLimiter(min_interval_seconds=0.2)

_HOLDINGS_COLUMNS = ["symbol", "quarter_end_date", "total_shares_held", "total_value", "num_filers"]
_TREND_COLUMNS = [*_HOLDINGS_COLUMNS, "change_from_prior_quarter"]


def _cache_dir(subdir: str) -> Path:
    return Path(get_settings().ingestion_cache_dir) / "edgar_13f" / subdir


def _headers() -> dict[str, str]:
    return {"User-Agent": get_settings().sec_edgar_user_agent}


def _month_end(year: int, month: int) -> date:
    return date(year, month, monthrange(year, month)[1])


def quarter_window_for(as_of: date) -> tuple[date, date]:
    """The SEC 13F bulk-file 3-month filing window containing `as_of`.

    Verified against the real, currently published set of window URLs
    (Mar-May, Jun-Aug, Sep-Nov, Dec-Feb, each a fixed calendar range) -- see
    module docstring for the one known historical exception this doesn't
    reproduce.
    """
    year, month = as_of.year, as_of.month
    if month == 12:
        start, end = (year, 12), (year + 1, 2)
    elif month in (1, 2):
        start, end = (year - 1, 12), (year, 2)
    elif month in (3, 4, 5):
        start, end = (year, 3), (year, 5)
    elif month in (6, 7, 8):
        start, end = (year, 6), (year, 8)
    else:
        start, end = (year, 9), (year, 11)
    return date(start[0], start[1], 1), _month_end(end[0], end[1])


def _prior_quarter_window(window: tuple[date, date]) -> tuple[date, date]:
    """The bulk-file window immediately preceding `window` in the fixed quarterly cycle."""
    start, _ = window
    return quarter_window_for(start - timedelta(days=1))


def latest_published_window(
    as_of: date, *, max_lookback: int = _MAX_WINDOW_LOOKBACK
) -> tuple[date, date] | None:
    """The newest window SEC has actually published at `as_of`, or `None`.

    **`quarter_window_for(today)` is never the right thing to ask for**, and
    asking for it is why this module produced zero rows for its entire life.
    That function returns the window *containing* today — the quarter currently
    being filed — and SEC publishes a window's bulk file only after the window
    closes. So the URL it names cannot exist on any day of any quarter, the
    fetch 404s, `refresh_institutional_ownership` logs the exception and returns
    0, the run reports "partial", and `institutional_ownership` stays empty
    forever while Smart Money silently runs on two of its three signals.

    Nor is "the previous window" a fix by itself. Publication lags the close by
    weeks: measured on 2026-09-05, five days after Jun-Aug closed, that window
    still 404ed while Mar-May served 200. Anything that assumes a fixed offset
    is the same bug with a different constant, so this *asks*, walking back one
    quarter at a time until a file is there.

    A locally cached ZIP counts as published without a network call, which keeps
    a warm cache offline-capable. `max_lookback` bounds the walk so an SEC
    outage costs a handful of HEADs rather than an unbounded crawl into 2024,
    and `None` (rather than a guess) is what the caller gets when nothing in
    that range is available.
    """
    window = quarter_window_for(as_of)
    for _ in range(max_lookback):
        if _local_zip_path(window).exists():
            return window
        _rate_limiter.wait()
        # Deliberately outside `get_breaker(_SOURCE).guard()`: a 404 here is an
        # answer, not a failure, and counting a run of them would open the
        # circuit and then block the download of the window that *was* found.
        if resource_exists(_bulk_zip_url(window), headers=_headers()):
            return window
        window = _prior_quarter_window(window)
    return None


def _format_window_date(d: date) -> str:
    return f"{d.day:02d}{_MONTH_ABBREVIATIONS[d.month - 1]}{d.year}"


def _bulk_zip_url(window: tuple[date, date]) -> str:
    start, end = window
    return _BULK_URL_TEMPLATE.format(start=_format_window_date(start), end=_format_window_date(end))


def _local_zip_path(window: tuple[date, date]) -> Path:
    start, end = window
    directory = _cache_dir("bulk_zips")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_format_window_date(start)}_{_format_window_date(end)}.zip"


def ensure_bulk_zip_downloaded(window: tuple[date, date]) -> Path:
    """Path to `window`'s bulk 13F ZIP, downloading it once if not already cached locally.

    A published window's data never changes, so this is cached on disk
    indefinitely rather than through `cached_json`/`cached_dataframe` (built
    around already-parsed Python objects, not a ~100MB binary blob). Warming
    this cache directory between runs matters exactly as much as the
    HuggingFace model cache does for the News Intelligence module
    (Section 6.10) -- the difference between re-downloading ~100MB and not.
    """
    path = _local_zip_path(window)
    if path.exists():
        return path
    _rate_limiter.wait()
    with get_breaker(_SOURCE).guard():
        content = get_bytes(
            _bulk_zip_url(window), headers=_headers(), timeout=_DOWNLOAD_TIMEOUT_SECONDS
        )
    path.write_bytes(content)
    return path


_NON_ALNUM_RE = re.compile(r"[^A-Z0-9&\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_CORP_SUFFIX_RE = re.compile(
    r"\b(?:INCORPORATED|INC|CORPORATION|CORP|COMPANIES|COMPANY|CO|HOLDINGS?|GROUP|LIMITED|LTD|PLC|LLC|LP)\s*$"
)


def _normalize_issuer_name(name: object) -> str:
    """Uppercase, punctuation- and corporate-suffix-stripped key for issuer-name matching.

    13F filers write issuer names inconsistently ("EBAY INC.", "APPLE INC")
    -- this collapses both sides (13F `NAMEOFISSUER` and the ticker
    universe's own `name` column) to a common key so they can be joined
    without a CUSIP mapping (module docstring).

    **Takes `object`, not `str`, because the real file contains blanks.** A
    filer may leave `NAMEOFISSUER` empty, and pandas reads that column as a
    float NaN, so this used to raise `AttributeError: 'float' object has no
    attribute 'strip'` on the very first chunk -- killing the whole quarter's
    ingest over a handful of unusable rows. An unnamed holding cannot be matched
    to a ticker by name under any circumstances, so the empty key it returns
    here is the honest answer: the row drops out at the `notna()` filter with
    every other unmatched one, and the other ~99.99% of the file still lands.

    This was invisible until the window bug above was fixed, because no real
    bulk file had ever been parsed.
    """
    if not isinstance(name, str):
        return ""
    normalized = _NON_ALNUM_RE.sub(" ", name.strip().upper())
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return _CORP_SUFFIX_RE.sub("", normalized).strip()


def fetch_quarterly_institutional_holdings(
    window: tuple[date, date], universe: pd.DataFrame
) -> pd.DataFrame:
    """Total institutional shares/value held per `universe` symbol, for one quarter.

    Streams `INFOTABLE.tsv` out of the (large) bulk ZIP in chunks rather than
    loading it whole, filtering to rows whose issuer name normalizes to one
    in `universe`, then joins `SUBMISSION.tsv` for each holding's reported
    period and filer. Rows whose `PERIODOFREPORT` isn't this window's
    dominant (most common) period are dropped before aggregating -- a bulk
    file can contain late amendments referencing an older period, and mixing
    a stale amendment's holdings into the current quarter's total would
    quietly corrupt it.

    Returns columns `symbol, quarter_end_date, total_shares_held,
    total_value, num_filers` -- empty (but correctly shaped) if nothing in
    `universe` was found.
    """
    normalized_to_symbol: dict[str, str] = {}
    for row in universe.itertuples(index=False):
        key = _normalize_issuer_name(str(row.name))
        if key:
            normalized_to_symbol.setdefault(key, str(row.symbol).strip().upper())

    if not normalized_to_symbol:
        # Nothing to match -- bail out before triggering the ~100MB download.
        return pd.DataFrame(columns=_HOLDINGS_COLUMNS)

    zip_path = ensure_bulk_zip_downloaded(window)

    matched_chunks: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("INFOTABLE.tsv") as info_file:
            for chunk in pd.read_csv(
                info_file,
                sep="\t",
                chunksize=_INFOTABLE_CHUNK_SIZE,
                usecols=["ACCESSION_NUMBER", "NAMEOFISSUER", "VALUE", "SSHPRNAMT"],
            ):
                chunk["symbol"] = (
                    chunk["NAMEOFISSUER"].map(_normalize_issuer_name).map(normalized_to_symbol)
                )
                matched = chunk[chunk["symbol"].notna()]
                if not matched.empty:
                    matched_chunks.append(matched)

        if not matched_chunks:
            return pd.DataFrame(columns=_HOLDINGS_COLUMNS)

        with archive.open("SUBMISSION.tsv") as submission_file:
            submissions = pd.read_csv(
                submission_file, sep="\t", usecols=["ACCESSION_NUMBER", "CIK", "PERIODOFREPORT"]
            )

    infotable = pd.concat(matched_chunks, ignore_index=True)
    merged = infotable.merge(submissions, on="ACCESSION_NUMBER", how="left")
    merged = merged.dropna(subset=["PERIODOFREPORT"])
    if merged.empty:
        return pd.DataFrame(columns=_HOLDINGS_COLUMNS)

    # Keep only the window's dominant reporting period -- see docstring.
    dominant_period = merged["PERIODOFREPORT"].mode().iat[0]
    merged = merged[merged["PERIODOFREPORT"] == dominant_period]

    aggregated = merged.groupby("symbol", as_index=False).agg(
        total_shares_held=("SSHPRNAMT", "sum"),
        total_value=("VALUE", "sum"),
        num_filers=("CIK", "nunique"),
    )
    aggregated["quarter_end_date"] = pd.to_datetime(dominant_period, format="%d-%b-%Y").date()
    return aggregated[_HOLDINGS_COLUMNS]


def fetch_institutional_ownership_trend(
    window: tuple[date, date], universe: pd.DataFrame
) -> pd.DataFrame:
    """`window`'s institutional holdings vs. the immediately preceding quarter's (Section 13).

    Each distinct window triggers its own one-time ~100MB download (cached
    indefinitely once fetched, so calling this repeatedly for the same
    window pair is cheap after the first call). A symbol held this quarter
    but not matched in the prior quarter gets `change_from_prior_quarter` =
    `NaN`, not a filled zero -- a real "no comparable prior figure" (which
    could be a genuinely new position, or a prior-quarter name-matching miss,
    module docstring) is left honestly unknown rather than silently assumed.
    """
    current = fetch_quarterly_institutional_holdings(window, universe)
    if current.empty:
        return pd.DataFrame(columns=_TREND_COLUMNS)

    # The prior quarter is for the *comparison* only, so it is allowed to be
    # missing: the oldest window the dataset carries has no predecessor, and a
    # transient failure on the second download should not throw away a
    # successful first one. Every symbol then gets `change_from_prior_quarter`
    # = NaN, which is already this function's documented "no comparable prior
    # figure" -- holdings and filer counts still land, and `smart_money`
    # renormalizes over the sub-signals that do have data.
    try:
        prior = fetch_quarterly_institutional_holdings(_prior_quarter_window(window), universe)
    except Exception:
        logger.warning(
            "13F: could not read the quarter before %s; storing holdings without a "
            "quarter-over-quarter change",
            window[0],
            exc_info=True,
        )
        prior = pd.DataFrame(columns=_HOLDINGS_COLUMNS)

    prior_shares = prior[["symbol", "total_shares_held"]].rename(
        columns={"total_shares_held": "_prior_shares_held"}
    )
    merged = current.merge(prior_shares, on="symbol", how="left")
    merged["change_from_prior_quarter"] = merged["total_shares_held"] - merged["_prior_shares_held"]
    return merged[_TREND_COLUMNS]
