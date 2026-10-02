"""Finding 39: `./run.sh` must start Streamlit with Arrow's system allocator.

Reproduced 2026-10-02 on this Mac with `run.sh` exactly as shipped: a fresh
`.venv-app` (pyarrow 25.0.0, Streamlit 1.59.1), the seven pages driven in a real
browser, and the server exited 139 (SIGSEGV) after about thirty page loads. The
crash report's faulting frames are `mi_heap_main` <- `mi_thread_init` in
`libarrow.2500.dylib` -- the mimalloc allocator Arrow bundles, initialising on a
fresh script thread -- the stack first diagnosed on 2026-08-27. To a visitor it is
a blank page and an idle terminal. `ARROW_DEFAULT_MEMORY_POOL=system` makes Arrow
use the system allocator and was the workaround in the notes; it was in no file.

The script is run for real here, in a copy with a stub `streamlit` that prints
its environment, so the test checks what Streamlit would actually be started
with rather than what the file happens to say.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_WATCHED = ("ARROW_DEFAULT_MEMORY_POOL", "DATABASE_URL", "PORTFOLIO_BACKEND")


def _environment_streamlit_gets(tmp_path: Path, **env: str) -> dict[str, str]:
    shutil.copy(ROOT / "run.sh", tmp_path / "run.sh")
    (tmp_path / "requirements.txt").write_text("streamlit\n")
    (tmp_path / "quantpulse_demo.db").write_bytes(b"already here, so nothing is fetched")
    venv_bin = tmp_path / ".venv-app" / "bin"
    venv_bin.mkdir(parents=True)
    # The stamp run.sh compares, so it skips the install.
    digest = hashlib.sha1((tmp_path / "requirements.txt").read_bytes()).hexdigest()
    (tmp_path / ".venv-app" / ".requirements-stamp").write_text(digest + "\n")
    stub = venv_bin / "streamlit"
    # Only the variables under test -- never the whole environment, which a failing
    # assertion would print into a CI log.
    stub.write_text(
        "#!/usr/bin/env bash\n"
        + "".join(f'printf "{name}=%s\\n" "${{{name}:-}}"\n' for name in _WATCHED)
    )
    stub.chmod(0o755)

    base = {key: value for key, value in os.environ.items() if key != "ARROW_DEFAULT_MEMORY_POOL"}
    result = subprocess.run(
        ["bash", str(tmp_path / "run.sh")],
        capture_output=True,
        text=True,
        env={**base, **env},
        check=True,
        timeout=60,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def test_streamlit_is_started_with_the_system_allocator(tmp_path: Path) -> None:
    env = _environment_streamlit_gets(tmp_path)
    assert env.get("ARROW_DEFAULT_MEMORY_POOL") == "system"
    # And the rest of what run.sh promises still arrives.
    assert env.get("DATABASE_URL") == "sqlite:///./quantpulse_demo.db"
    assert env.get("PORTFOLIO_BACKEND") == "session"


def test_a_pool_the_user_chose_is_left_alone(tmp_path: Path) -> None:
    env = _environment_streamlit_gets(tmp_path, ARROW_DEFAULT_MEMORY_POOL="jemalloc")
    assert env.get("ARROW_DEFAULT_MEMORY_POOL") == "jemalloc"
