import { useEffect } from "react";
import { Disclaimer } from "./components/Common";
import { Mark } from "./components/Mark";
import { Link, navigate, useMatch, usePath } from "./lib/router";
import Dashboard from "./pages/Dashboard";
import Screener from "./pages/Screener";
import StockDetail from "./pages/StockDetail";
import TrackRecord from "./pages/TrackRecord";
import Glossary from "./pages/Glossary";

const NAV = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/screener", label: "Screener" },
  { to: "/track-record", label: "Track Record" },
  { to: "/glossary", label: "Glossary" },
];

function Routes() {
  const path = usePath();
  const stockMatch = useMatch("/stocks/:symbol");

  // "/" is an alias for the dashboard; redirect rather than render it at two
  // URLs, so a shared link is always the canonical one.
  useEffect(() => {
    if (path === "/") navigate("/dashboard");
  }, [path]);

  if (stockMatch) return <StockDetail symbol={stockMatch.symbol} />;
  switch (path) {
    case "/":
    case "/dashboard":
      return <Dashboard />;
    case "/screener":
      return <Screener />;
    case "/track-record":
      return <TrackRecord />;
    case "/glossary":
      return <Glossary />;
    default:
      // Say what is missing and offer the way out. "Page not found." is the
      // placeholder voice the rest of this app does not use anywhere else.
      return (
        <>
          <h1>No such page</h1>
          <p className="standfirst">
            There is nothing at <code>{path}</code>. The four pages are in the bar above;{" "}
            <Link to="/screener">the Screener</Link> is where most routes lead.
          </p>
        </>
      );
  }
}

export default function App() {
  const path = usePath();
  return (
    <div className="app">
      {/*
        Skip link, and it is first in the DOM because that is the whole point:
        it must be the first thing a keyboard lands on. Without it, reaching the
        page content from the address bar means tabbing the brand link and every
        nav item on every navigation — and on the Screener, leaving the content
        again means tabbing past 1,509 controls inside the table.

        Visible only when focused. A skip link that is always visible is a
        design decision this layout has not made; one that never becomes visible
        is a trap, because a sighted keyboard user cannot tell what they have
        focused.
      */}
      <a href="#content" className="skiplink">
        Skip to content
      </a>
      <header className="topbar">
        <Link to="/dashboard" className="brand">
          <Mark />
          QuantPulse
        </Link>
        <nav aria-label="Main">
          {NAV.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              className={path === item.to ? "navlink active" : "navlink"}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </header>

      {/*
        `tabIndex={-1}` so the skip link's target can actually receive focus.
        Without it the browser scrolls to the landmark but leaves focus on the
        link, and the next Tab returns to the nav — the link appears to work
        and does nothing for the keyboard user it exists for.
      */}
      <main className="content" id="content" tabIndex={-1}>
        <Routes />
      </main>

      <Disclaimer />
    </div>
  );
}
