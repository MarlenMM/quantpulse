import { type FormEvent, useMemo, useState } from "react";

import { EmptyState, ErrorBox, LoadingTable, Metric, RatingChip } from "../components/Common";
import { api } from "../lib/api";
import { downloadCsv } from "../lib/csv";
import { formatPercent, formatPrice, formatScore, formatSignedPercent } from "../lib/format";
import {
  analyse,
  buildLotBook,
  holdingTerm,
  POSITION_THRESHOLD,
  VAR_CONFIDENCE,
  type Bars,
  type Inputs,
  type Rating,
  type Transaction,
} from "../lib/portfolio";
import {
  EMPTY,
  EXAMPLE,
  transactionsFromCsv,
  transactionsToCsv,
  usePortfolio,
} from "../lib/portfolioStore";
import { Link } from "../lib/router";
import type { ScreenerRow, StockDetail } from "../lib/types";
import { useApi } from "../lib/useApi";

/**
 * The Portfolio page of the public demo (finding 33).
 *
 * Half the product's pitch -- a portfolio manager -- was reachable only by
 * cloning the repository. This is the part of it a static site can honestly
 * offer: a transaction log kept in this browser, and everything the engine
 * derives from one that needs only published data -- FIFO lots and realized
 * gains, current value and P/L, the per-holding Add/Hold/Trim/Sell with its
 * reason, concentration and sector gaps, and the risk of today's mix.
 * `lib/portfolio.ts` computes all of it and is pinned to the engine by a golden
 * file, so these are the full app's numbers, not a second opinion.
 *
 * What it does not do, and says so: the three optimisers, the rebalancing
 * trade list and the history panel (they need the engine's solvers or a
 * dividend history the demo does not publish), stock splits after a purchase,
 * and any storage beyond this browser.
 */

type Loaded = Record<string, StockDetail | null>;

const ACTION_LABEL: Record<string, string> = {
  add: "▲ Add",
  hold: "■ Hold",
  trim: "▼ Trim",
  sell: "▼▼ Sell",
};

function money(value: number | null | undefined): string {
  return formatPrice(value);
}

function signedMoney(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${value >= 0 ? "+" : "−"}${formatPrice(Math.abs(value))}`;
}

/** The published close on `date`, or the last one before it. */
function closeOnOrBefore(detail: StockDetail, date: string): number | null {
  let found: number | null = null;
  for (const bar of detail.prices) {
    if (bar.date > date) break;
    found = bar.close;
  }
  return found;
}

function toBars(detail: StockDetail): Bars {
  return detail.prices.map((bar) => [bar.date, bar.close, bar.adj_close ?? bar.close]);
}

/** Top five by composite in each sector -- what the full app offers for a gap. */
function sectorCandidates(rows: ScreenerRow[]): Record<string, string[]> {
  const bySector: Record<string, ScreenerRow[]> = {};
  for (const row of rows) {
    if (!row.sector || row.composite_score === null) continue;
    (bySector[row.sector] ??= []).push(row);
  }
  return Object.fromEntries(
    Object.entries(bySector).map(([sector, list]) => [
      sector,
      [...list]
        .sort((a, b) => (b.composite_score ?? 0) - (a.composite_score ?? 0))
        .slice(0, 5)
        .map((row) => row.symbol),
    ]),
  );
}

export default function Portfolio() {
  const [state, setState] = usePortfolio();
  const screener = useApi(() => api.screener(), []);

  // The lot book needs no prices, so it says which symbols to fetch.
  const book = useMemo(() => {
    try {
      return { value: buildLotBook(state.transactions), error: null as string | null };
    } catch (error) {
      return { value: null, error: (error as Error).message };
    }
  }, [state.transactions]);
  const openSymbols = book.value ? Object.keys(book.value.open_lots).sort() : [];
  const stocks = useApi<Loaded>(
    () =>
      Promise.all(
        openSymbols.map((symbol) =>
          api
            .stock(symbol)
            .then((detail): [string, StockDetail | null] => [symbol, detail])
            .catch((): [string, StockDetail | null] => [symbol, null]),
        ),
      ).then((pairs) => Object.fromEntries(pairs)),
    [openSymbols.join(",")],
  );

  const rows = screener.data?.rows ?? [];
  const bySymbol = useMemo(() => new Map(rows.map((row) => [row.symbol, row])), [rows]);

  const result = useMemo(() => {
    if (!book.value || !stocks.data || !screener.data) return null;
    const loaded = stocks.data;
    const inputs: Inputs = {
      cash: state.cash,
      transactions: state.transactions,
      prices: Object.fromEntries(
        Object.entries(loaded)
          .filter((entry): entry is [string, StockDetail] => entry[1] !== null && entry[1].prices.length > 0)
          .map(([symbol, detail]) => [symbol, toBars(detail)]),
      ),
      sectors: Object.fromEntries(
        openSymbols.map((s) => [s, bySymbol.get(s)?.sector ?? loaded[s]?.summary.sector ?? null]),
      ),
      ratings: Object.fromEntries(
        openSymbols
          .map((s) => [s, bySymbol.get(s)?.rating] as const)
          .filter((entry): entry is [string, Rating] => Boolean(entry[1])),
      ),
      sector_candidates: sectorCandidates(rows),
    };
    return analyse(inputs);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [book.value, stocks.data, screener.data, state.cash, state.transactions]);

  const asOf = useMemo(() => {
    const dates = Object.values(stocks.data ?? {})
      .flatMap((detail) => (detail?.prices.length ? [detail.prices[detail.prices.length - 1].date] : []))
      .sort();
    return dates[dates.length - 1] ?? null;
  }, [stocks.data]);

  const empty = state.transactions.length === 0;

  return (
    <>
      <h1>Portfolio</h1>
      <p className="standfirst">
        A portfolio kept in this browser, scored the way the screener scores the market: FIFO
        lots, profit and loss, a suggested action per holding, concentration, and the risk of
        what you own.
      </p>
      <p className="callout callout-note">
        <strong>This browser only.</strong> Nothing is sent anywhere — the site is static files
        and has no server to send it to. It is not synced to other devices, and clearing this
        site&rsquo;s data clears it. Export a CSV to keep it; the same file imports into the full
        app&rsquo;s Portfolio Manager, which adds the optimisers, the rebalancing trade list and
        the history panel.
      </p>

      <Controls state={state} setState={setState} />

      {book.error ? (
        <p className="callout callout-error">The transaction log does not add up: {book.error}</p>
      ) : null}

      {empty ? (
        <EmptyState>
          Nothing here yet. Add a trade below, import a CSV, or load the example portfolio —
          five well-known names across five sectors, there to show the analysis working, not as
          a suggestion.
        </EmptyState>
      ) : screener.error ? (
        <ErrorBox error={screener.error} />
      ) : !result ? (
        <LoadingTable what="your holdings" />
      ) : (
        <Analysis result={result} bySymbol={bySymbol} asOf={asOf} bookLots={book.value!.open_lots} />
      )}

      <Ledger state={state} setState={setState} realized={result?.realized ?? []} />
      <AddTrade state={state} setState={setState} rows={rows} />
    </>
  );
}

type Result = ReturnType<typeof analyse>;

function Analysis({
  result,
  bySymbol,
  asOf,
  bookLots,
}: {
  result: Result;
  bySymbol: Map<string, ScreenerRow>;
  asOf: string | null;
  bookLots: ReturnType<typeof buildLotBook>["open_lots"];
}) {
  const positions = Object.entries(result.positions) as [
    string,
    {
      shares: number;
      cost_basis: number;
      average_cost: number;
      current_price: number | null;
      market_value: number | null;
      unrealized_gain: number | null;
      weight: number;
    },
  ][];
  const invested = positions.reduce((s, [, p]) => s + (p.market_value ?? 0), 0);
  const cost = positions.reduce((s, [, p]) => s + p.cost_basis, 0);
  const unrealized = positions.reduce((s, [, p]) => s + (p.unrealized_gain ?? 0), 0);
  const realized = result.realized.reduce((s, g) => s + g.gain, 0);
  const risk = result.risk;
  const stale = positions.filter(([, p]) => p.current_price === null).map(([s]) => s);

  return (
    <>
      <section className="card lede">
        <h2 className="h-lede">Holdings</h2>
        <div className="metrics">
          <Metric label="Total value" value={money(result.total_value)} hint="holdings plus cash" />
          <Metric label="Invested" value={money(invested)} hint={`cost ${money(cost)}`} />
          <Metric
            label="Unrealized P/L"
            value={signedMoney(unrealized)}
            hint={cost > 0 ? formatSignedPercent(unrealized / cost) : undefined}
            term="Unrealized P/L"
          />
          <Metric label="Realized P/L" value={signedMoney(realized)} hint="from sells, FIFO" term="FIFO" />
        </div>
        {stale.length ? (
          <p className="callout callout-warn">
            No published price for {stale.join(", ")} — it is outside the S&amp;P 500 universe this
            demo publishes, so it is left out of the value, weights and risk rather than guessed.
          </p>
        ) : null}
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Symbol</th>
                <th scope="col" className="num">Shares</th>
                <th scope="col" className="num">Avg cost</th>
                <th scope="col" className="num">Last close</th>
                <th scope="col" className="num">Value</th>
                <th scope="col" className="num">Weight</th>
                <th scope="col" className="num">Unrealized</th>
                <th scope="col">Rating</th>
                <th scope="col">Suggested</th>
              </tr>
            </thead>
            <tbody>
              {positions.map(([symbol, p]) => {
                const rec = (result.recommendations as Record<string, { action: string; reason: string }>)[symbol];
                return (
                  <tr key={symbol}>
                    <td>
                      <Link to={`/stocks/${symbol}`}>
                        <span className="ticker">{symbol}</span>
                      </Link>
                    </td>
                    <td className="num">{formatScore(p.shares, p.shares % 1 === 0 ? 0 : 4)}</td>
                    <td className="num">{money(p.average_cost)}</td>
                    <td className="num">{money(p.current_price)}</td>
                    <td className="num">{money(p.market_value)}</td>
                    <td className="num">{formatPercent(p.weight)}</td>
                    <td className="num">
                      {signedMoney(p.unrealized_gain)}{" "}
                      <span className="muted">
                        {p.unrealized_gain !== null ? formatSignedPercent(p.unrealized_gain / p.cost_basis) : ""}
                      </span>
                    </td>
                    <td>
                      <RatingChip rating={bySymbol.get(symbol)?.rating ?? null} />
                    </td>
                    <td title={rec?.reason}>{rec ? ACTION_LABEL[rec.action] : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <details>
          <summary>Why each suggestion, and the open tax lots</summary>
          <ul>
            {Object.entries(result.recommendations as Record<string, { action: string; reason: string }>).map(
              ([symbol, rec]) => (
                <li key={symbol}>
                  <strong>{symbol}</strong> — {ACTION_LABEL[rec.action]}. {rec.reason}
                </li>
              ),
            )}
          </ul>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Symbol</th>
                  <th scope="col">Bought</th>
                  <th scope="col" className="num">Shares</th>
                  <th scope="col" className="num">Cost basis</th>
                  <th scope="col">Held</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(bookLots).flatMap(([symbol, lots]) =>
                  lots.map((lot) => (
                    <tr key={`${symbol}-${lot.purchase_date}-${lot.shares}`}>
                      <td>{symbol}</td>
                      <td>{lot.purchase_date}</td>
                      <td className="num">{formatScore(lot.shares, lot.shares % 1 === 0 ? 0 : 4)}</td>
                      <td className="num">{money(lot.cost_basis)}</td>
                      <td>{asOf ? (holdingTerm(lot.purchase_date, asOf) === "long" ? "over a year" : "a year or less") : "—"}</td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
          <p className="note">
            Suggestions are the screener&rsquo;s rating mapped to an action — Buy or Strong Buy to
            Add, Hold to Hold, Sell to Trim, Strong Sell to Sell — except that nothing at or above{" "}
            {formatPercent(POSITION_THRESHOLD, 0)} of the portfolio is suggested for adding to.
            Holding periods are descriptive, never tax advice.
          </p>
        </details>
      </section>

      <section className="block">
        <h2>
          Concentration
        </h2>
        <div className="metrics">
          <Metric
            label="Position HHI"
            value={formatScore(result.concentration.position_hhi, 3)}
            hint={
              result.concentration.position_effective_count
                ? `as diversified as ${formatScore(result.concentration.position_effective_count, 1)} equal positions`
                : undefined
            }
            term="Herfindahl index"
          />
          <Metric
            label="Sector HHI"
            value={formatScore(result.concentration.sector_hhi, 3)}
            hint={
              result.concentration.sector_effective_count
                ? `as diversified as ${formatScore(result.concentration.sector_effective_count, 1)} equal sectors`
                : undefined
            }
            term="Herfindahl index"
          />
        </div>
        {result.concentration.warnings.length ? (
          <ul className="callout callout-warn">
            {result.concentration.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        ) : (
          <p className="note">No position or sector above the 15% concentration guideline.</p>
        )}
        {result.rebalance_reasons.length ? (
          <p className="note">
            A rebalance is worth considering: {result.rebalance_reasons.join("; ")}. The full
            app builds the trade list.
          </p>
        ) : null}
        <div className="split-even">
          <div className="tablewrap">
            <table>
              <caption className="sr-only">Invested value by sector</caption>
              <thead>
                <tr>
                  <th scope="col">Sector</th>
                  <th scope="col" className="num">Share of invested value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.sector_weights)
                  .sort((a, b) => b[1] - a[1])
                  .map(([sector, weight]) => (
                    <tr key={sector}>
                      <td>{sector}</td>
                      <td className="num">{formatPercent(weight)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <details>
            <summary>{result.sector_gaps.length} sectors you hold nothing in</summary>
            <ul>
              {result.sector_gaps.map((gap) => (
                <li key={gap}>{gap}</li>
              ))}
            </ul>
          </details>
        </div>
      </section>

      <section className="block">
        <h2>Risk of today&rsquo;s mix</h2>
        <p className="note">
          Hypothetical: your current weights, cash included, applied to the last{" "}
          {risk.n_observations} trading days of published prices — not the path your holdings
          actually took. The full app measures over 420 calendar days of stored history, so its
          figures cover a longer window than this demo publishes.
        </p>
        <div className="metrics">
          <Metric label="Volatility (ann.)" value={formatPercent(risk.volatility)} term="Volatility" />
          <Metric
            label={`1-day VaR (${formatPercent(VAR_CONFIDENCE, 0)})`}
            value={formatPercent(risk.var)}
            hint={risk.var !== null ? `about ${money(risk.var * result.total_value)} on a bad day` : "needs 100 days of shared history"}
            term="Value at Risk"
          />
          <Metric
            label="Expected shortfall"
            value={formatPercent(risk.expected_shortfall)}
            term="Expected shortfall"
          />
          <Metric label="Max drawdown" value={formatPercent(risk.max_drawdown)} term="Max drawdown" />
          <Metric
            label="Avg correlation"
            value={formatScore(risk.average_correlation, 2)}
            hint={risk.average_correlation === null ? "needs 30 shared days per pair" : undefined}
            term="Correlation"
          />
        </div>
        {Object.keys(risk.correlations).length > 1 ? (
          <div className="tablewrap">
            <table>
              <caption className="sr-only">Pairwise correlation of daily returns</caption>
              <thead>
                <tr>
                  <th scope="col">
                    <span className="sr-only">Symbol</span>
                  </th>
                  {Object.keys(risk.correlations).map((s) => (
                    <th key={s} scope="col" className="num">
                      {s}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(risk.correlations).map(([a, row]) => (
                  <tr key={a}>
                    <th scope="row">{a}</th>
                    {Object.entries(row).map(([b, value]) => (
                      <td key={b} className="num">
                        {formatScore(value, 2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>
    </>
  );
}

function Controls({
  state,
  setState,
}: {
  state: { cash: number; transactions: Transaction[] };
  setState: (next: { cash: number; transactions: Transaction[] }) => void;
}) {
  const [message, setMessage] = useState<string | null>(null);

  async function importFile(file: File) {
    try {
      const transactions = transactionsFromCsv(await file.text());
      buildLotBook(transactions); // refuse a log that sells what it never held
      setState({ ...state, transactions });
      setMessage(`Imported ${transactions.length} transactions from ${file.name}.`);
    } catch (error) {
      setMessage(`Not imported: ${(error as Error).message}`);
    }
  }

  return (
    <div className="controls">
      <button
        type="button"
        onClick={() => {
          if (state.transactions.length === 0 || window.confirm("Replace your portfolio with the example?")) {
            setState(EXAMPLE);
            setMessage(null);
          }
        }}
      >
        Load the example portfolio
      </button>
      <button
        type="button"
        disabled={state.transactions.length === 0}
        onClick={() => downloadCsv("quantpulse_transactions.csv", transactionsToCsv(state.transactions))}
      >
        Export transactions as CSV
      </button>
      <label>
        Import a transactions CSV
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void importFile(file);
            e.target.value = "";
          }}
        />
      </label>
      <label>
        Cash
        <input
          type="number"
          min={0}
          step="0.01"
          value={state.cash}
          onChange={(e) => {
            const cash = Number(e.target.value);
            if (Number.isFinite(cash) && cash >= 0) setState({ ...state, cash });
          }}
        />
      </label>
      <button
        type="button"
        disabled={state.transactions.length === 0 && state.cash === 0}
        onClick={() => {
          if (window.confirm("Clear every transaction and the cash balance from this browser?")) {
            setState(EMPTY);
            setMessage(null);
          }
        }}
      >
        Clear the portfolio
      </button>
      {message ? (
        <p className="note" role="status">
          {message}
        </p>
      ) : null}
    </div>
  );
}

function Ledger({
  state,
  setState,
  realized,
}: {
  state: { cash: number; transactions: Transaction[] };
  setState: (next: { cash: number; transactions: Transaction[] }) => void;
  realized: Result["realized"];
}) {
  if (state.transactions.length === 0) return null;
  return (
    <section className="block">
      <h2>Transactions</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Date</th>
              <th scope="col">Symbol</th>
              <th scope="col">Action</th>
              <th scope="col" className="num">Shares</th>
              <th scope="col" className="num">Price</th>
              <th scope="col">
                <span className="sr-only">Remove</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {state.transactions.map((t, index) => (
              <tr key={`${index}-${t.symbol}-${t.date}`}>
                <td>{t.date}</td>
                <td>{t.symbol}</td>
                <td>{t.action === "buy" ? "Buy" : "Sell"}</td>
                <td className="num">{formatScore(t.shares, t.shares % 1 === 0 ? 0 : 4)}</td>
                <td className="num">{money(t.price)}</td>
                <td>
                  <button
                    type="button"
                    className="iconbutton"
                    aria-label={`Remove the ${t.action} of ${t.symbol} on ${t.date}`}
                    onClick={() => {
                      const next = state.transactions.filter((_, i) => i !== index);
                      try {
                        buildLotBook(next);
                        setState({ ...state, transactions: next });
                      } catch (error) {
                        window.alert(`Removing it would leave a sell with nothing to sell: ${(error as Error).message}`);
                      }
                    }}
                  >
                    ✕
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {realized.length ? (
        <details>
          <summary>Realized gains, lot by lot (FIFO)</summary>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Sold</th>
                  <th scope="col">Symbol</th>
                  <th scope="col">Bought</th>
                  <th scope="col" className="num">Shares</th>
                  <th scope="col" className="num">Gain</th>
                  <th scope="col">Held</th>
                </tr>
              </thead>
              <tbody>
                {realized.map((g, i) => (
                  <tr key={i}>
                    <td>{g.sale_date}</td>
                    <td>{g.symbol}</td>
                    <td>{g.purchase_date}</td>
                    <td className="num">{formatScore(g.shares, g.shares % 1 === 0 ? 0 : 4)}</td>
                    <td className="num">{signedMoney(g.gain)}</td>
                    <td>{g.term === "long" ? "over a year" : "a year or less"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      ) : null}
    </section>
  );
}

function AddTrade({
  state,
  setState,
  rows,
}: {
  state: { cash: number; transactions: Transaction[] };
  setState: (next: { cash: number; transactions: Transaction[] }) => void;
  rows: ScreenerRow[];
}) {
  const [symbol, setSymbol] = useState("");
  const [action, setAction] = useState<"buy" | "sell">("buy");
  const [shares, setShares] = useState("");
  const [date, setDate] = useState("");
  const [price, setPrice] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setProblem(null);
    const ticker = symbol.trim().toUpperCase();
    const count = Number(shares);
    if (!ticker) return setProblem("Enter a symbol.");
    if (!(Number.isFinite(count) && count > 0)) return setProblem("Shares must be a number above zero.");
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return setProblem("Choose the trade date.");

    let paid = price.trim() === "" ? NaN : Number(price);
    if (price.trim() === "") {
      // Blank means "that day's close", from the same published prices the
      // page values holdings with.
      setBusy(true);
      try {
        const detail = await api.stock(ticker);
        const close = closeOnOrBefore(detail, date);
        if (close === null) {
          setBusy(false);
          return setProblem(
            `No published ${ticker} price on or before ${date} — the demo carries about a year of prices. Enter the price you paid.`,
          );
        }
        paid = close;
      } catch {
        setBusy(false);
        return setProblem(`${ticker} is not in the published universe. Enter the price you paid.`);
      }
      setBusy(false);
    }
    if (!(Number.isFinite(paid) && paid > 0)) return setProblem("Price must be a number above zero.");

    const next = [...state.transactions, { symbol: ticker, action, shares: count, price: paid, date }];
    try {
      buildLotBook(next);
    } catch (error) {
      return setProblem((error as Error).message);
    }
    setState({ ...state, transactions: next });
    setShares("");
    setPrice("");
  }

  return (
    <section className="block">
      <h2>Add a trade</h2>
      <form className="controls" onSubmit={(e) => void submit(e)}>
        <label>
          Symbol
          <input
            list="portfolio-symbols"
            value={symbol}
            placeholder="e.g. NVDA"
            autoComplete="off"
            onChange={(e) => setSymbol(e.target.value)}
          />
        </label>
        {/* Outside the label: inside it, every option's text became part of the
            input's accessible name -- "Symbol Agilent Technologies Apple Inc. ..." */}
        <datalist id="portfolio-symbols">
          {rows.map((row) => (
            <option key={row.symbol} value={row.symbol}>
              {row.name ?? row.symbol}
            </option>
          ))}
        </datalist>
        <label>
          Buy or sell
          <select value={action} onChange={(e) => setAction(e.target.value as "buy" | "sell")}>
            <option value="buy">Buy</option>
            <option value="sell">Sell</option>
          </select>
        </label>
        <label>
          Shares
          <input type="number" min={0} step="any" value={shares} onChange={(e) => setShares(e.target.value)} />
        </label>
        <label>
          Date
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
        </label>
        <label>
          Price per share (blank = that day&rsquo;s close)
          <input type="number" min={0} step="any" value={price} onChange={(e) => setPrice(e.target.value)} />
        </label>
        <button type="submit" disabled={busy}>
          {busy ? "Looking up the close…" : "Add trade"}
        </button>
      </form>
      {problem ? (
        <p className="callout callout-error" role="alert">
          {problem}
        </p>
      ) : null}
    </section>
  );
}
