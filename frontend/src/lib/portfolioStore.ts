/**
 * Where the demo's portfolio lives: this browser's `localStorage` (finding 33).
 *
 * The site is static files and the API is read-only by design (ADR 4.5), so
 * there is no server to keep a visitor's holdings on -- the same position the
 * watchlist is in, stated the same way on the page: this browser only, not
 * synced, gone if site data is cleared. The CSV below is the full app's own
 * transaction format (`portfolio/holdings.to_csv` / `from_csv`), so a portfolio
 * built here can be exported and imported into the Streamlit Portfolio Manager.
 *
 * The example and the column list are copies of the engine's constants; the
 * golden file pins both (`tests/portfolio.spec.ts`).
 */
import { useCallback, useEffect, useState } from "react";

import type { Transaction } from "./portfolio";

const KEY = "quantpulse.portfolio";

export interface PortfolioState {
  cash: number;
  transactions: Transaction[];
}

export const EMPTY: PortfolioState = { cash: 0, transactions: [] };

/** `holdings.CSV_COLUMNS`. */
export const CSV_COLUMNS = ["symbol", "action", "shares", "price", "date"] as const;

/** `holdings.example_state()`: well-known, multi-sector names; scaffolding, not advice. */
export const EXAMPLE: PortfolioState = {
  cash: 5000,
  transactions: [
    { symbol: "AAPL", action: "buy", shares: 25, price: 180.5, date: "2024-03-14" },
    { symbol: "MSFT", action: "buy", shares: 15, price: 405.2, date: "2024-05-02" },
    { symbol: "JNJ", action: "buy", shares: 30, price: 152.1, date: "2023-11-08" },
    { symbol: "XOM", action: "buy", shares: 40, price: 104.75, date: "2024-01-22" },
    { symbol: "JPM", action: "buy", shares: 20, price: 188.3, date: "2024-07-09" },
  ],
};

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

function isTransaction(value: unknown): value is Transaction {
  const t = value as Transaction;
  return (
    typeof t === "object" &&
    t !== null &&
    typeof t.symbol === "string" &&
    t.symbol.length > 0 &&
    (t.action === "buy" || t.action === "sell") &&
    Number.isFinite(t.shares) &&
    t.shares > 0 &&
    Number.isFinite(t.price) &&
    t.price > 0 &&
    typeof t.date === "string" &&
    ISO_DATE.test(t.date)
  );
}

export function readPortfolio(): PortfolioState {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return EMPTY;
    const parsed = JSON.parse(raw) as Partial<PortfolioState>;
    // Filtered rather than trusted: the value is editable by hand.
    const transactions = Array.isArray(parsed.transactions)
      ? parsed.transactions.filter(isTransaction)
      : [];
    const cash = Number.isFinite(parsed.cash) && (parsed.cash as number) >= 0 ? (parsed.cash as number) : 0;
    return { cash, transactions };
  } catch {
    return EMPTY;
  }
}

function writePortfolio(state: PortfolioState): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // A blocked or full store: the page still works for this visit.
  }
}

/** `[state, set]` for this browser, persisted on every change. */
export function usePortfolio(): [PortfolioState, (next: PortfolioState) => void] {
  const [state, setState] = useState<PortfolioState>(() => readPortfolio());
  useEffect(() => writePortfolio(state), [state]);
  const set = useCallback((next: PortfolioState) => setState(next), []);
  return [state, set];
}

// ------------------------------------------------------------------------ CSV

/** The transaction log in the full app's CSV format. */
export function transactionsToCsv(transactions: Transaction[]): string {
  const rows = transactions.map((t) => [t.symbol, t.action, t.shares, t.price, t.date].join(","));
  return [CSV_COLUMNS.join(","), ...rows].join("\r\n") + "\r\n";
}

function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quoted) {
      if (char === '"' && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else if (char === '"') quoted = false;
      else current += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") {
      cells.push(current);
      current = "";
    } else current += char;
  }
  cells.push(current);
  return cells;
}

function positiveNumber(text: string, field: string): number {
  const value = Number(text);
  if (text.trim() === "" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${field} must be a finite number > 0, got '${text}'`);
  }
  return value;
}

/**
 * `holdings.from_csv`: the header in any case and column order; a bad row is
 * rejected by its line number rather than half-imported.
 */
export function transactionsFromCsv(text: string): Transaction[] {
  const lines = text.split(/\r?\n/);
  const header = (lines[0] ?? "").trim();
  if (!header) throw new Error("CSV appears to be empty");
  const names = splitCsvLine(header).map((name) => name.trim().toLowerCase());
  const missing = CSV_COLUMNS.filter((column) => !names.includes(column));
  if (missing.length) throw new Error(`CSV is missing required column(s): ${missing.join(", ")}`);

  const parsed: Transaction[] = [];
  lines.slice(1).forEach((line, index) => {
    const cells = splitCsvLine(line).map((cell) => cell.trim());
    if (!cells.some((cell) => cell)) return;
    const row = Object.fromEntries(names.map((name, i) => [name, cells[i] ?? ""]));
    try {
      const action = row.action.toLowerCase();
      if (action !== "buy" && action !== "sell") {
        throw new Error(`action must be 'buy' or 'sell', got '${row.action}'`);
      }
      if (!ISO_DATE.test(row.date) || Number.isNaN(Date.parse(row.date))) {
        throw new Error(`Invalid isoformat string: '${row.date}'`);
      }
      parsed.push({
        symbol: row.symbol.toUpperCase(),
        action,
        shares: positiveNumber(row.shares, "shares"),
        price: positiveNumber(row.price, "price"),
        date: row.date,
      });
    } catch (error) {
      throw new Error(`row ${index + 2} is invalid: ${(error as Error).message}`);
    }
  });
  return parsed;
}
