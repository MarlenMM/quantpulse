/**
 * CSV export for the SPA (Section 10's "download as CSV").
 *
 * Streamlit has this on three pages via `st.download_button`; the SPA had it on
 * none, and it is the front end most visitors ever see. There is no server to
 * ask for a file — the site is pre-rendered JSON on GitHub Pages — so the CSV
 * is built from the rows already in the browser and handed over as a blob.
 */

/** Quote a cell for RFC 4180: double the quotes, wrap anything ambiguous. */
function cell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  // A bare comma, quote, newline or carriage return would otherwise end the
  // field early — and a name like "Alphabet, Inc." is not exotic.
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/**
 * Rows to CSV text, with a header taken from `columns`.
 *
 * `columns` is explicit rather than derived from the first row's keys: the
 * header order is what a reader sees, and deriving it would silently reorder
 * the file whenever the API added a field.
 */
export function toCsv<T extends object>(
  rows: readonly T[],
  columns: readonly { key: keyof T & string; label: string }[],
): string {
  const header = columns.map((c) => cell(c.label)).join(",");
  const body = rows.map((row) => columns.map((c) => cell(row[c.key])).join(","));
  // CRLF per RFC 4180, and a trailing newline so the last row is terminated —
  // some spreadsheet importers drop an unterminated final line.
  return [header, ...body].join("\r\n") + "\r\n";
}

/**
 * Hand `text` to the browser as a download named `filename`.
 *
 * A blob URL rather than a `data:` URI: a screener export of 503 rows is well
 * past the length some browsers accept in a URL, and a truncated CSV that still
 * opens is worse than one that fails.
 */
export function downloadCsv(filename: string, text: string): void {
  // A BOM, so Excel reads the file as UTF-8 rather than as the local codepage.
  // Without it a company name with an accent arrives mangled, which looks like
  // a data bug rather than an encoding one.
  const blob = new Blob(["﻿", text], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Revoking immediately can cancel the download in some browsers; a tick is
  // enough for the click to have been handed off.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
