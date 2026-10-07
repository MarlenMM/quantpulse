"""Serve the code that was pushed, not the code the process first loaded (point 46).

Streamlit Community Cloud applies every push to a running app's files and keeps
the process. Streamlit drops changed modules from `sys.modules` only for
sessions connected when the files change, so a push that lands between visits
leaves the old `lib.*` and `quantpulse.*` modules loaded while each page script
is read fresh. Measured on 2026-09-25: after the finding-34 push the hosted
Stock Detail page raised `ImportError: cannot import name 'format_edge_cell'
from 'lib.format'` until the app was rebooted.

Every page calls `refresh_changed_code()` before its other `lib` imports. It
compares the modification times of the repository's Python sources with the
snapshot taken the last time it ran, and drops every loaded module whose source
changed (and the attribute its package holds, which `from pkg import mod` reads
before `sys.modules`), so the page's own imports load the pushed code. The cost
is a stat of each source file, a few milliseconds.

A function call rather than an import side effect: a module cannot remove
itself from `sys.modules` while it is being imported (the import machinery looks
it up again when the file finishes), so an import cannot be made to run on
every page load.

The snapshot lives on this module, which is never dropped. The first call in a
process only records it, so a process that was already stale before this
existed needed one reboot.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
#: Sources whose modules are shared across page runs: the app's helpers and the
#: engine. Page scripts themselves are re-read by Streamlit on every run.
ROOTS = (_APP / "lib", _APP.parent / "src" / "quantpulse")

Snapshot = dict[str, float]


def snapshot(roots: Iterable[Path]) -> Snapshot:
    """Modification time of every Python source under `roots`."""
    times: Snapshot = {}
    for root in roots:
        for path in Path(root).rglob("*.py"):
            try:
                times[str(path.resolve())] = path.stat().st_mtime
            except OSError:
                continue
    return times


def drop_stale_modules(roots: Iterable[Path], before: Snapshot) -> tuple[list[str], Snapshot]:
    """If any source under `roots` changed since `before`, drop every loaded module there.

    Every one, not only the changed files (finding 49). A module that did not
    change still holds the objects it imported: `lib.data` kept the *old*
    `persistence` after a push that changed only engine modules, so the pushed
    reader never ran -- served stale until a reboot, found by running 48's
    migration check on a real server. Re-importing the whole tree once per push
    costs a page load a little time; serving half-old code costs correctness.

    Returns (names dropped, new snapshot).
    """
    roots = [Path(r).resolve() for r in roots]
    now = snapshot(roots)
    changed = {path for path, mtime in now.items() if before.get(path) != mtime}
    if not changed:
        return [], now
    dropped: list[str] = []
    for name, module in list(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if not source or not any(Path(source).resolve().is_relative_to(r) for r in roots):
            continue
        if name == __name__:
            continue
        del sys.modules[name]
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, child, None) is module:
            delattr(parent, child)
        dropped.append(name)
    return sorted(dropped), now


_last: Snapshot | None = None


def refresh_changed_code() -> list[str]:
    """Drop the repository's modules if any source changed since the last call. Returns names."""
    global _last
    if _last is None:
        _last = snapshot(ROOTS)
        return []
    dropped, _last = drop_stale_modules(ROOTS, _last)
    return dropped
