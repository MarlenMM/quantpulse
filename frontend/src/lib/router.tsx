/**
 * A ~50-line router over the History API, replacing `react-router-dom`.
 *
 * Not a preference — a dependency decision. Every published react-router
 * version currently sits inside a high-severity advisory range (there is no
 * 8.x, and the newest 7.x carries an RSC-mode CSRF advisory while older ones
 * carry a dozen more, including RCE). This SPA needs exactly three things from
 * a router: match five paths, read one URL parameter, and intercept link
 * clicks. Carrying a large dependency with no clean release to get them is a
 * poor trade in a project whose dependency discipline (Section 29) is part of
 * the pitch.
 *
 * What it deliberately does NOT do: nested routes, loaders, code splitting,
 * server rendering. If this app ever needs those, take the dependency back —
 * by then it will presumably have a patched release.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

const PathContext = createContext<string>("/");

export function navigate(to: string): void {
  if (to !== window.location.pathname) {
    window.history.pushState({}, "", to);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }
}

export function Router({ children }: { children: ReactNode }) {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  return <PathContext.Provider value={path}>{children}</PathContext.Provider>;
}

export function usePath(): string {
  return useContext(PathContext);
}

/**
 * An anchor that navigates client-side but is still a real `<a href>`.
 *
 * Keeping the href means middle-click, ctrl-click and "open in new tab" work,
 * and screen readers announce it as a link — all of which a `<div onClick>`
 * would silently break. Modified clicks fall through to the browser.
 */
export function Link({
  to,
  children,
  className,
}: {
  to: string;
  children: ReactNode;
  className?: string;
}) {
  const path = usePath();
  const onClick = useCallback(
    (event: React.MouseEvent<HTMLAnchorElement>) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
      event.preventDefault();
      navigate(to);
    },
    [to],
  );
  const isActive = path === to;
  return (
    <a
      href={to}
      onClick={onClick}
      className={className}
      aria-current={isActive ? "page" : undefined}
    >
      {children}
    </a>
  );
}

/**
 * Match the current path against `"/stocks/:symbol"`-style patterns.
 * Returns the first match's params, or null.
 */
export function useMatch(pattern: string): Record<string, string> | null {
  const path = usePath();
  return useMemo(() => {
    const patternParts = pattern.split("/").filter(Boolean);
    const pathParts = path.split("/").filter(Boolean);
    if (patternParts.length !== pathParts.length) return null;
    const params: Record<string, string> = {};
    for (let i = 0; i < patternParts.length; i += 1) {
      const expected = patternParts[i];
      if (expected.startsWith(":")) params[expected.slice(1)] = decodeURIComponent(pathParts[i]);
      else if (expected !== pathParts[i]) return null;
    }
    return params;
  }, [path, pattern]);
}
