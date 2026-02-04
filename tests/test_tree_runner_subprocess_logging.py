from __future__ import annotations

import os
from pathlib import Path


def test_run_python_logged_writes_stdout_and_stderr(tmp_path: Path) -> None:
    from agentic_experimentation.tree_runner import _run_python_logged

    log_path = tmp_path / "subprocess.log"
    rc = _run_python_logged(
        repo_root=Path(".").resolve(),
        args=[
            "-c",
            "import sys; print('hello-out'); print('hello-err', file=sys.stderr)",
        ],
        env=dict(os.environ),
        log_path=log_path,
        echo=False,
    )
    assert rc == 0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    assert "[stdout] hello-out" in text
    assert "[stderr] hello-err" in text

