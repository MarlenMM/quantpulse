from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base. Tables land here starting in Phase 1 (Section 13)."""


class Ticker(Base):
    """Universe definition. `is_active` flags current-vs-removed without deleting history."""

    __tablename__ = "tickers"

    symbol: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(100))
    exchange: Mapped[str | None] = mapped_column(String(20))
    asset_type: Mapped[str] = mapped_column(String(10), default="equity")
    is_active: Mapped[bool] = mapped_column(default=True)


class IndexMembershipHistory(Base):
    """Point-in-time index membership, for survivorship-bias-aware backtests (Section 5, 22).

    Table only in Phase 1 — populating it correctly during the cold-start
    backfill is Phase 1's Opus-owned half of the work.
    """

    __tablename__ = "index_membership_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    index_name: Mapped[str] = mapped_column(String(50))
    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"))
    added_date: Mapped[date] = mapped_column(Date)
    removed_date: Mapped[date | None] = mapped_column(Date)


class PriceHistory(Base):
    """Raw OHLCV price data, keyed so a re-run can find each symbol's last stored date."""

    __tablename__ = "price_history"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)


class FundamentalsSnapshot(Base):
    """Point-in-time fundamentals — append-only by (symbol, as_of_date), never overwritten."""

    __tablename__ = "fundamentals_snapshot"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    pe: Mapped[float | None] = mapped_column(Float)
    pb: Mapped[float | None] = mapped_column(Float)
    ps: Mapped[float | None] = mapped_column(Float)
    peg: Mapped[float | None] = mapped_column(Float)
    eps: Mapped[float | None] = mapped_column(Float)
    revenue_growth: Mapped[float | None] = mapped_column(Float)
    debt_equity: Mapped[float | None] = mapped_column(Float)
    roe: Mapped[float | None] = mapped_column(Float)
    roa: Mapped[float | None] = mapped_column(Float)
    fcf: Mapped[float | None] = mapped_column(Float)
    div_yield: Mapped[float | None] = mapped_column(Float)
    # Sector-specific substitutes (e.g. FFO for REITs, Section 7.2) — populated in Phase 3.
    sector_specific_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class AnalystConsensus(Base):
    """Wall Street analyst rating counts + mean price target, point-in-time."""

    __tablename__ = "analyst_consensus"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    strong_buy: Mapped[int] = mapped_column(default=0)
    buy: Mapped[int] = mapped_column(default=0)
    hold: Mapped[int] = mapped_column(default=0)
    sell: Mapped[int] = mapped_column(default=0)
    strong_sell: Mapped[int] = mapped_column(default=0)
    mean_price_target: Mapped[float | None] = mapped_column(Float)


class MacroIndicator(Base):
    """Raw FRED macro series (Section 5) — one row per indicator per date."""

    __tablename__ = "macro_indicators"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    indicator_name: Mapped[str] = mapped_column(String(50), primary_key=True)
    value: Mapped[float] = mapped_column(Float)


class PatternSignal(Base):
    """Detected chart/candlestick patterns (Section 13).

    Shared by candlestick detection (Phase 2, native library detectors) and
    the geometric chart-pattern algorithms (Phase 2's Opus half, e.g.
    head-and-shoulders) -- both normalize to this same shape.
    """

    __tablename__ = "pattern_signals"
    __table_args__ = (UniqueConstraint("symbol", "date", "pattern_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"))
    date: Mapped[date] = mapped_column(Date)
    pattern_type: Mapped[str] = mapped_column(String(50))
    direction: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[float] = mapped_column(Float)


class RefreshLog(Base):
    """Pipeline health monitoring for each nightly/cold-start run."""

    __tablename__ = "refresh_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(50))
    run_timestamp: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20))
    rows_updated: Mapped[int] = mapped_column(default=0)
