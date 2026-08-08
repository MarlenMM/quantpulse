"""The full list of US-listed securities, from Nasdaq's own public symbol directory.

`wikipedia_client` answers "what is in the S&P 500", which is the universe the
nightly job scores. This answers a different question -- "what exists at all" --
so a visitor can search for, and have the app analyse, a company that is not one
of those 500.

The two files below are the canonical listing directory Nasdaq publishes for the
whole US market: `nasdaqlisted.txt` for Nasdaq itself and `otherlisted.txt` for
NYSE, NYSE American, Cboe and the rest. They are plain pipe-delimited text, need
no key, no account and no registration, and together return about 13,000 rows in
a couple of seconds -- which is the entire reason this is the source rather than
a commercial reference API with a free tier that can be withdrawn.

**Cataloguing is not the same as covering.** A row here means the symbol exists
and can be searched for; it does not mean any price, score or fundamental has
ever been fetched for it. That distinction is carried by the `coverage` column
on `tickers` and is the thing that keeps a 13,000-row catalogue from turning
into a 13,000-ticker nightly job.
"""

from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path

import pandas as pd

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_dataframe
from quantpulse.ingestion.http import get_text

_NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
_OTHER_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

# Both files end with a "File Creation Time" trailer that is not a security.
_TRAILER_PREFIX = "File Creation"

# `otherlisted.txt`'s single-letter exchange codes.
_EXCHANGE_NAMES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}

# Suffix characters the directory uses for share classes and for instruments
# that are not ordinary equity (warrants, units, rights, preferreds). A symbol
# carrying one is kept in the catalogue but is not a candidate for the ranked
# universe -- see `is_ordinary_equity`.
_NON_ORDINARY_MARKERS = ("$", "+", "^", "~", "=")


def _parse(text: str, symbol_column: str, exchange_column: str | None) -> pd.DataFrame:
    # The trailer is stripped from the *text*, not from the parsed frame. It
    # does not carry the same number of fields as the header, so leaving it in
    # makes pandas raise before any row-level filter could run -- the file is
    # rejected wholesale over its own footer.
    body = "\n".join(
        line for line in text.splitlines() if line and not line.startswith(_TRAILER_PREFIX)
    )
    # `na_filter=False` is not a nicety. Nano Labs trades as **NA**, and pandas'
    # default NA-value list turns that string into a missing value -- so the
    # symbol arrives as NaN, the row inserts a NULL primary key, and the whole
    # catalogue sync dies on a NOT NULL constraint. `dtype=str` does not prevent
    # it; only switching NA detection off does.
    frame = pd.read_csv(io.StringIO(body), sep="|", dtype=str, na_filter=False)
    parsed = pd.DataFrame(
        {
            "symbol": frame[symbol_column].astype(str).str.strip(),
            "name": frame["Security Name"].astype(str).str.strip(),
            "is_etf": frame.get("ETF", pd.Series("N", index=frame.index)).eq("Y"),
            "is_test": frame.get("Test Issue", pd.Series("N", index=frame.index)).eq("Y"),
        }
    )
    if exchange_column is not None:
        parsed["exchange"] = frame[exchange_column].map(_EXCHANGE_NAMES).fillna("Other")
    else:
        parsed["exchange"] = "Nasdaq"
    return parsed


def _fetch_nasdaq() -> pd.DataFrame:
    return _parse(get_text(_NASDAQ_URL), "Symbol", None)


def _fetch_other() -> pd.DataFrame:
    return _parse(get_text(_OTHER_URL), "ACT Symbol", "Exchange")


def is_ordinary_equity(frame: pd.DataFrame) -> pd.Series:
    """Rows that are plain company stock, as opposed to a fund or a derivative.

    The directory mixes ordinary shares in with ETFs, warrants, units, rights
    and preferred series. All of them are worth having in a catalogue -- someone
    typing "SPY" should find it -- but only ordinary equity is a sensible
    candidate for a *ranking*, because the composite score is built from company
    fundamentals, analyst estimates and company news, none of which a warrant
    has.
    """
    ordinary = ~frame["is_etf"] & ~frame["is_test"]
    for marker in _NON_ORDINARY_MARKERS:
        ordinary &= ~frame["symbol"].str.contains(marker, regex=False)
    # "XYZ Warrant", "... Units", "... Rights", "... % Preferred ..."
    return ordinary & ~frame["name"].str.contains(
        r"\b(?:warrant|unit|right|preferred|depositary|note|debenture)s?\b",
        case=False,
        regex=True,
        na=False,
    )


def fetch_us_listings() -> pd.DataFrame:
    """Every currently-listed US security, as `[symbol, name, exchange, asset_type]`.

    Test issues are dropped -- they are exchange plumbing, not tradeable. Nothing
    else is filtered out here: `asset_type` records what each row is and lets the
    caller decide, which keeps the "what exists" question separate from the "what
    do we rank" one.
    """
    cache_dir = Path(get_settings().ingestion_cache_dir) / "listings"
    ttl = timedelta(days=1)
    nasdaq = cached_dataframe("nasdaq_listed", _fetch_nasdaq, cache_dir, ttl=ttl)
    other = cached_dataframe("other_listed", _fetch_other, cache_dir, ttl=ttl)

    listings = pd.concat([nasdaq, other], ignore_index=True)
    listings = listings[~listings["is_test"]]
    # Nasdaq's directory lists a handful of symbols on both files; the Nasdaq
    # file wins because it is the primary listing venue for those rows.
    listings = listings.drop_duplicates(subset="symbol", keep="first")

    ordinary = is_ordinary_equity(listings)
    listings = listings.assign(
        asset_type=pd.Series("other", index=listings.index)
        .mask(listings["is_etf"], "etf")
        .mask(ordinary, "equity")
    )

    # Data providers use '-' where the directory uses '.' (BRK.B -> BRK-B), the
    # same normalisation `wikipedia_client` applies, so a catalogue row and a
    # ranked row for the same company cannot end up as two different symbols.
    listings["symbol"] = listings["symbol"].str.replace(".", "-", regex=False)

    # Belt and braces behind the NA fix above: a row with no symbol is not a
    # security, and it is better to lose one malformed line than the file.
    usable = listings["symbol"].str.strip().ne("") & listings["name"].str.strip().ne("")

    return (
        listings.loc[usable, ["symbol", "name", "exchange", "asset_type"]]
        .sort_values("symbol")
        .reset_index(drop=True)
    )
