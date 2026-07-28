import { useEffect } from "react";
import { Disclaimer } from "./components/Common";
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
      return <p className="muted">Page not found.</p>;
  }
}

export default function App() {
  const path = usePath();
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span aria-hidden="true">📈</span> QuantPulse
        </div>
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
