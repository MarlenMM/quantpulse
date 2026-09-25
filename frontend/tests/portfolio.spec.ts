import { readFileSync } from "node:fs";
import { join } from "node:path";

import { expect, test } from "@playwright/test";

import { analyse, holdingTerm, percent0, type Inputs, type Transaction } from "../src/lib/portfolio";
import {
  CSV_COLUMNS,
  EXAMPLE,
  transactionsFromCsv,
  transactionsToCsv,
} from "../src/lib/portfolioStore";

/**
 * The demo's portfolio computes what the engine computes (finding 33).
 *
 * `tests/fixtures/portfolio-golden.json` holds two synthetic portfolios and the
 * engine's own results for them; `tests/unit/test_portfolio_golden.py` requires
 * the engine to still produce those results, and this requires the TypeScript
 * port to produce them too. Neither side can change alone. No browser: these
 * are pure functions, and `analyse()` is exactly what the page renders.
 */
const golden = JSON.parse(
  readFileSync(join(process.cwd(), "tests", "fixtures", "portfolio-golden.json"), "utf8"),
) as {
  holding_terms: [string, string, "short" | "long"][];
  example: { cash: number; transactions: Transaction[] };
  csv_columns: string[];
  engine_csv: string;
  scenarios: { inputs: Inputs & { name: string }; expected: unknown }[];
};

function close(actual: unknown, wanted: unknown, path: string): void {
  if (wanted === null || typeof wanted !== "object") {
    if (typeof wanted === "number" && !Number.isInteger(wanted)) {
      expect(typeof actual, path).toBe("number");
      const tolerance = Math.max(1e-12, Math.abs(wanted) * 1e-9);
      expect(Math.abs((actual as number) - wanted), `${path}: ${actual} vs ${wanted}`).toBeLessThanOrEqual(
        tolerance,
      );
    } else if (typeof wanted === "number") {
      expect(Math.abs((actual as number) - wanted), `${path}: ${actual} vs ${wanted}`).toBeLessThanOrEqual(
        Math.max(1e-12, Math.abs(wanted) * 1e-9),
      );
    } else {
      expect(actual, path).toEqual(wanted);
    }
    return;
  }
  if (Array.isArray(wanted)) {
    expect(Array.isArray(actual), path).toBe(true);
    expect((actual as unknown[]).length, `${path}.length`).toBe(wanted.length);
    wanted.forEach((w, i) => close((actual as unknown[])[i], w, `${path}[${i}]`));
    return;
  }
  const a = actual as Record<string, unknown>;
  expect(Object.keys(a).sort(), `${path} keys`).toEqual(Object.keys(wanted).sort());
  for (const key of Object.keys(wanted)) close(a[key], (wanted as Record<string, unknown>)[key], `${path}.${key}`);
}

for (const { inputs, expected } of golden.scenarios) {
  test(`the portfolio matches the engine: ${inputs.name}`, () => {
    close(analyse(inputs), expected, "analyse");
  });
}

test("the one-year rule agrees with the engine at its edges", () => {
  for (const [purchase, asOf, term] of golden.holding_terms) {
    expect(holdingTerm(purchase, asOf), `${purchase} -> ${asOf}`).toBe(term);
  }
});

test("percentages round half to even, as the engine's messages do", () => {
  // Python's f"{0.125:.0%}" is "12%"; JavaScript's toFixed would say "13%".
  expect(percent0(0.125)).toBe("12%");
  expect(percent0(0.135)).toBe("14%");
  expect(percent0(0.15)).toBe("15%");
  expect(percent0(0.2349)).toBe("23%");
});

test("selling more than is held is refused, as the engine refuses it", () => {
  expect(() =>
    analyse({
      cash: 0,
      transactions: [
        { symbol: "A", action: "buy", shares: 1, price: 10, date: "2026-01-02" },
        { symbol: "A", action: "sell", shares: 2, price: 10, date: "2026-01-05" },
      ],
      prices: {},
      sectors: {},
      ratings: {},
      sector_candidates: {},
    }),
  ).toThrow(/cannot sell/);
});

test("the example portfolio and the CSV columns are the engine's", () => {
  expect(EXAMPLE).toEqual(golden.example);
  expect([...CSV_COLUMNS]).toEqual(golden.csv_columns);
});

test("a CSV exported by the full app imports here unchanged", () => {
  expect(transactionsFromCsv(golden.engine_csv)).toEqual(golden.scenarios[0].inputs.transactions);
});

test("the demo's own export imports back unchanged", () => {
  const transactions = golden.scenarios[0].inputs.transactions;
  expect(transactionsFromCsv(transactionsToCsv(transactions))).toEqual(transactions);
});

test("a bad row is rejected by its line number, not half-imported", () => {
  const text = "Symbol,Action,Shares,Price,Date\r\nAAPL,buy,1,10,2026-01-02\r\nMSFT,hold,1,10,2026-01-02\r\n";
  expect(() => transactionsFromCsv(text)).toThrow("row 3 is invalid: action must be 'buy' or 'sell'");
  expect(() => transactionsFromCsv("symbol,action\r\n")).toThrow(/missing required column/);
  expect(() => transactionsFromCsv("symbol,action,shares,price,date\r\nA,buy,nan,1,2026-01-02"))
    .toThrow(/shares must be a finite number/);
});
