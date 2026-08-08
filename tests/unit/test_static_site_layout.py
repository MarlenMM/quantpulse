"""The pre-rendered site covers every request the SPA can make.

On GitHub Pages a request the generator never wrote is a 404 in the browser --
a blank section, or a whole blank page, on the public demo. Two ways that
happens, and one check for each:

* **The client asks for an endpoint nobody generated.** Every method on the
  `api` object in `frontend/src/lib/api.ts` must map onto something
  `scripts/build_static_site.py` writes.
* **The client asks for a different limit.** `api.regime(90)` and a generated
  `regime__limit-90.json` agree today; changing the 90 in the client without
  changing the generator produces exactly one missing file. The call sites are
  parsed out of the `.tsx` pages and compared.

The naming rule itself lives in both languages -- `static_path()` here,
`staticPath()` in `api.ts` -- and only the Python half is pinned below. The
TypeScript half is checked end to end instead, by `.github/workflows/pages.yml`
loading the built site in a real browser before publishing it: if the two ever
disagreed, every request would 404 and that check fails loudly. A unit test
that re-implemented the rule a third time to compare against would only be
pinning a third copy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from build_static_site import FIXED_REQUESTS, PROFILE_REQUESTS, static_path

REPO = Path(__file__).resolve().parents[2]
API_CLIENT = REPO / "frontend" / "src" / "lib" / "api.ts"
PAGES = REPO / "frontend" / "src" / "pages"

# Requests whose arguments are runtime values rather than literals, so the
# generator expands them from the database instead of from a fixed list.
DYNAMIC = {
    "stock": "/stocks/{symbol} -- one file per scored symbol",
    "screener": "/screener -- one file per investor profile",
    "screenerAbsolute": "/screener/absolute -- one file per investor profile",
}

# Methods the client exposes but no page calls. Generated anyway (they cost one
# small file each) so a page that starts using one is not a 404.
UNUSED = {"universe"}


def _client_methods() -> dict[str, tuple[str, tuple[str, ...]]]:
    """`{method: (path, param names)}`, read out of the api.ts client itself."""
    source = API_CLIENT.read_text()
    body = source.split("export const api = {", 1)[1]
    methods: dict[str, tuple[str, tuple[str, ...]]] = {}
    for name, path, params in re.findall(
        r"(\w+):\s*\([^)]*\)\s*=>\s*(?:\n\s*)?request<[^>]+>\(\s*[`\"]([^`\"]+)[`\"]"
        r"(?:\s*,\s*\{([^}]*)\})?",
        body,
    ):
        keys = tuple(part.split(":")[0].strip() for part in params.split(",") if part.strip())
        methods[name] = (path, keys)
    return methods


def _literal_calls() -> dict[str, list[str]]:
    """`{method: [argument text]}` for every `api.<method>(...)` in the pages."""
    calls: dict[str, list[str]] = {}
    for page in sorted(PAGES.glob("*.tsx")):
        for name, args in re.findall(r"api\.(\w+)\(([^()]*)\)", page.read_text()):
            calls.setdefault(name, []).append(args.strip())
    return calls


def test_the_client_exposes_the_methods_this_test_reasons_about() -> None:
    """A parse failure here would make every assertion below vacuously true."""
    methods = _client_methods()
    assert set(methods) >= {"health", "glossary", "screener", "stock", "regime"}
    assert methods["regime"] == ("/regime", ("limit",))
    assert methods["stock"][0].startswith("/stocks/")


def test_every_client_method_is_generated() -> None:
    generated_paths = {path for path, _ in FIXED_REQUESTS} | set(PROFILE_REQUESTS)
    for name, (path, _) in _client_methods().items():
        if name in DYNAMIC:
            continue
        assert path in generated_paths, (
            f"api.{name}() requests {path}, which scripts/build_static_site.py "
            "never writes -- it would 404 on the published site"
        )


def test_generated_limits_match_the_call_sites() -> None:
    methods = _client_methods()
    generated = {(path, tuple(sorted((params or {}).items()))) for path, params in FIXED_REQUESTS}
    calls = _literal_calls()

    for name, argument_lists in calls.items():
        if name in DYNAMIC:
            continue
        path, keys = methods[name]
        for args in argument_lists:
            if not args:
                continue  # defaults; covered by the generated entry for this path
            if not re.fullmatch(r"-?\d+", args):
                continue  # not a plain literal
            assert keys, f"api.{name}({args}) passes an argument the client sends nowhere"
            expected = (path, ((keys[0], int(args)),))
            assert expected in generated, (
                f"a page calls api.{name}({args}) but the generator writes "
                f"{[p for p, _ in generated if p == path]} for {path} -- "
                "update FIXED_REQUESTS in scripts/build_static_site.py"
            )


def test_call_sites_were_actually_found() -> None:
    """Guards the regex above: no matches would make the previous test vacuous."""
    calls = _literal_calls()
    assert calls.get("regime") == ["90"]
    assert calls.get("news") == ["6"]
    assert calls.get("backtest") == ["20"]
    assert calls.get("ratingChanges") == ["8"]


@pytest.mark.parametrize(
    ("path", "params", "expected"),
    [
        ("/health", None, "health.json"),
        ("/regime", {"limit": 90}, "regime__limit-90.json"),
        ("/screener/absolute", {"profile": "income"}, "screener__absolute__profile-income.json"),
        ("/stocks/BRK.B", None, "stocks__BRK.B.json"),
        # Sorted, so one request always names one file whatever order the
        # client happened to build its parameter object in.
        ("/x", {"b": 2, "a": 1}, "x__a-1__b-2.json"),
    ],
)
def test_static_path_naming(path: str, params: dict[str, object] | None, expected: str) -> None:
    assert static_path(path, params) == expected


def test_unused_client_methods_are_still_generated() -> None:
    """Cheap insurance: a page that starts calling one must not 404."""
    generated_paths = {path for path, _ in FIXED_REQUESTS}
    methods = _client_methods()
    for name in UNUSED:
        assert methods[name][0] in generated_paths
