"""The full US listing directory, parsed into a catalogue.

The risk here is not parsing -- it is classification. Nasdaq's directory mixes
ordinary shares in with ETFs, warrants, units, rights and preferred series, and
treating a warrant as a company is how a "stock screener" ends up ranking a
derivative on fundamentals it does not have.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from quantpulse.ingestion import listing_client

# Real rows, trimmed to the columns the parser reads. Both files carry a
# "File Creation Time" trailer that is not a security.
NASDAQ_FILE = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
    "Round Lot Size|ETF|NextShares\n"
    """AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|Q|N|N|100|Y|N
ZZZZT|Nasdaq Test Stock|G|Y|N|100|N|N
RKLBW|Rocket Lab Corporation - Warrant|Q|N|N|100|N|N
NA|Nano Labs Ltd - Class A Ordinary Shares|Q|N|N|100|N|N
File Creation Time: 0808202617:30|||||||
"""
)

OTHER_FILE = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|"
    "Round Lot Size|Test Issue|NASDAQ Symbol\n"
    """A|Agilent Technologies, Inc. Common Stock|N|A|N|100|N|A
BRK.B|Berkshire Hathaway Inc. Class B Common Stock|N|BRK.B|N|100|N|BRK.B
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
BAC$B|Bank of America Corporation Depositary Shares|N|BAC.PRB|N|100|N|BAC-B
File Creation Time: 0808202617:30||||||||
"""
)


def _fetch(url: str, **kwargs: object) -> str:
    return NASDAQ_FILE if "nasdaqlisted" in url else OTHER_FILE


def _listings() -> pd.DataFrame:
    # `cached_dataframe` would otherwise persist between tests and across runs.
    with (
        patch.object(listing_client, "get_text", _fetch),
        patch.object(listing_client, "cached_dataframe", lambda key, fn, *a, **k: fn()),
    ):
        return listing_client.fetch_us_listings()


def test_both_files_are_merged() -> None:
    listings = _listings()
    assert {"AAPL", "QQQ", "A", "SPY"} <= set(listings["symbol"])


def test_test_issues_are_dropped() -> None:
    """Exchange plumbing, not something anyone can buy."""
    assert "ZZZZT" not in set(_listings()["symbol"])


def test_the_trailer_line_is_not_a_security() -> None:
    symbols = set(_listings()["symbol"])
    assert not any(s.startswith("File Creation") for s in symbols)


def test_funds_and_derivatives_are_catalogued_but_not_called_equity() -> None:
    """Someone typing "SPY" should find it; the ranking should not score it.

    A warrant or a depositary share has no earnings, no analyst estimates and no
    company news, so a composite score built from those inputs would be
    meaningless rather than merely thin.
    """
    listings = _listings().set_index("symbol")
    assert listings.loc["AAPL", "asset_type"] == "equity"
    assert listings.loc["A", "asset_type"] == "equity"
    assert listings.loc["QQQ", "asset_type"] == "etf"
    assert listings.loc["SPY", "asset_type"] == "etf"
    assert listings.loc["RKLBW", "asset_type"] == "other"  # warrant, by name
    assert listings.loc["BAC$B", "asset_type"] == "other"  # depositary, by symbol marker


def test_symbols_are_normalised_the_way_price_providers_spell_them() -> None:
    """`BRK.B` here has to be the same row as `BRK-B` in the ranked universe.

    `wikipedia_client` already applies this substitution. If only one of the two
    did, the same company would occupy two catalogue entries and the ranked one
    would be unreachable from search.
    """
    symbols = set(_listings()["symbol"])
    assert "BRK-B" in symbols
    assert "BRK.B" not in symbols


def test_exchange_codes_are_resolved_to_names() -> None:
    listings = _listings().set_index("symbol")
    assert listings.loc["A", "exchange"] == "NYSE"
    assert listings.loc["SPY", "exchange"] == "NYSE Arca"
    assert listings.loc["AAPL", "exchange"] == "Nasdaq"


def test_the_ticker_NA_survives_pandas_na_handling() -> None:
    """Nano Labs trades as **NA**, and pandas reads that string as a missing value.

    Not hypothetical and not cosmetic: it was the first thing the live file did.
    The symbol arrived as NaN, went in as a NULL primary key, and took the whole
    catalogue sync down on a NOT NULL constraint -- which then poisoned the
    shared session and killed the entire nightly run. `dtype=str` does not
    prevent it; only turning NA detection off does.
    """
    listings = _listings()
    assert "NA" in set(listings["symbol"])
    assert listings.set_index("symbol").loc["NA", "name"].startswith("Nano Labs")
    assert not listings["symbol"].isna().any()
