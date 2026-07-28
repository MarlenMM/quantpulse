/**
 * Minimal fetch-on-mount hook.
 *
 * Deliberately not TanStack Query or similar: this app makes a handful of GETs
 * against data that changes once a night, so a caching/invalidation library
 * would be more machinery than the problem has. The `cancelled` flag prevents
 * a state update after unmount, which is the one real bug a hand-rolled hook
 * usually has.
 */
import { useEffect, useState } from "react";

export interface AsyncState<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
}

export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    error: null,
    loading: true,
  });

  useEffect(() => {
    let cancelled = false;
    setState({ data: null, error: null, loading: true });
    fetcher()
      .then((data) => {
        if (!cancelled) setState({ data, error: null, loading: false });
      })
      .catch((error: Error) => {
        if (!cancelled) setState({ data: null, error, loading: false });
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
