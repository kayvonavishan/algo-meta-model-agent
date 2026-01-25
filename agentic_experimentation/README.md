Agentic experimentation scaffold for improving the meta model.

Quick start
1) Edit `agentic_experimentation/agent_config.json` and point `ideas_file` at your idea list.
2) Put your ideas in either:
   - a single text file (blank-line separated blocks), or
   - a directory of `*.md`/`*.txt` files (one idea per file), e.g. `agentic_experimentation/ideas/`.
3) Set your API key env var if you want a hosted LLM (for example `OPENAI_API_KEY`).
4) If using the Codex MCP multi-agent runner, install dependencies:
   - `pip install openai-agents openai`
   - Ensure Node is available (for `npx`) and Codex CLI can run (the default config uses `npx -y codex mcp-server`).
   - If using `coder_backend: "cli"` / `"cli_session"`, ensure Codex CLI is runnable via `codex` (or configure `codex_cli` / `CODEX_CLI_CMD`).
   - If Codex reports `WRITE_BLOCKED` in a git worktree, upgrade Codex CLI and ensure the runner passes a working root (newer runner versions write `codex_cli_cmd_round_*.txt` for debugging).
   - If Codex CLI says it is "Logged in using ChatGPT", it currently runs in a forced `read-only` sandbox for local repos; run `codex logout` then login with an API key via `$env:OPENAI_API_KEY | codex login --with-api-key`.
5) Refresh the baseline snapshot:
   `python agentic_experimentation/agent_runner.py --config agentic_experimentation/agent_config.json --refresh-baseline`
6) Run iterations (defaults to one per idea):
   `python agentic_experimentation/agent_runner.py --config agentic_experimentation/agent_config.json`
   Use `--iterations N` to run only the first N ideas.
7) Multi-agent option (planner -> coder (Codex MCP edits) -> reviewer -> tests -> sweep):
   `python agentic_experimentation/multi_agent_runner.py --config agentic_experimentation/agent_config.json`

Notes
- Worktrees keep your main repo clean; each iteration runs in its own temp worktree.
- `adaptive_vol_momentum.py` is used as the sweep entry point in the sample config.
- The scoring hook is intentionally minimal; update `scoring_hooks.py` once you decide on the metric.
- Idea generation is file-driven only; the LLM is used for planning/review and to drive Codex MCP edits.
