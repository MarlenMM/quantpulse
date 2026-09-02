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

      <main className="content">
        <Routes />
      </main>

      <Disclaimer />
    </div>
  );
}
