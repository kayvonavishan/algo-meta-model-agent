Agentic experimentation scaffold for improving the meta model.

Quick start
1) Edit `agentic_experimentation/agent_config.json` and point `ideas_file` at your idea list.
2) Put your ideas in either:
   - a single text file (blank-line separated blocks), or
   - a directory of `*.md`/`*.txt` files (one idea per file), e.g. `agentic_experimentation/ideas/`.
3) Set your API key env var if you want a hosted LLM (for example `OPENAI_API_KEY`).
4) Refresh the baseline snapshot:
   `python agentic_experimentation/agent_runner.py --config agentic_experimentation/agent_config.json --refresh-baseline`
5) Run iterations (defaults to one per idea):
   `python agentic_experimentation/agent_runner.py --config agentic_experimentation/agent_config.json`
   Use `--iterations N` to run only the first N ideas.
6) Multi-agent option (coordinator → coder → reviewer → tests → sweep):
   `python agentic_experimentation/multi_agent_runner.py --config agentic_experimentation/agent_config.json`

Notes
- Worktrees keep your main repo clean; each iteration runs in its own temp worktree.
- `adaptive_vol_momentum.py` is used as the sweep entry point in the sample config.
- The scoring hook is intentionally minimal; update `scoring_hooks.py` once you decide on the metric.
- Idea generation is file-driven only; the LLM is used solely for patch generation.
