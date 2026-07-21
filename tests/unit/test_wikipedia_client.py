from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from quantpulse.ingestion import wikipedia_client


def _fake_settings(tmp_path: Path) -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    return settings


def test_fetch_sp500_constituents_normalizes_columns(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "Symbol": ["BRK.B", "AAPL"],
            "Security": ["Berkshire Hathaway", "Apple Inc."],
            "GICS Sector": ["Financials", "Technology"],
            "GICS Sub-Industry": ["Multi-Sector Holdings", "Consumer Electronics"],
            "Headquarters Location": ["Omaha, NE", "Cupertino, CA"],
            "Date added": ["1957-03-04", "1982-11-30"],
            "CIK": [1067983, 320193],
            "Founded": [1839, 1976],
        }
    )

    with (
        patch(
            "quantpulse.ingestion.wikipedia_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch(
            "quantpulse.ingestion.wikipedia_client.pd.read_html", return_value=[raw]
        ) as mock_read_html,
    ):
        df = wikipedia_client.fetch_sp500_constituents()

    assert list(df.columns) == [
        "symbol",
        "name",
        "sector",
        "industry",
        "exchange",
        "asset_type",
        "is_active",
    ]
    assert df.loc[df["name"] == "Berkshire Hathaway", "symbol"].iloc[0] == "BRK-B"
    assert (df["asset_type"] == "equity").all()
    assert df["is_active"].all()
    mock_read_html.assert_called_once()


def test_fetch_sp500_constituents_uses_cache_on_second_call(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "Symbol": ["AAPL"],
            "Security": ["Apple Inc."],
            "GICS Sector": ["Technology"],
            "GICS Sub-Industry": ["Consumer Electronics"],
        }
    )

    with (
        patch(
            "quantpulse.ingestion.wikipedia_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch(
            "quantpulse.ingestion.wikipedia_client.pd.read_html", return_value=[raw]
        ) as mock_read_html,
    ):
        wikipedia_client.fetch_sp500_constituents()
        wikipedia_client.fetch_sp500_constituents()

    mock_read_html.assert_called_once()
