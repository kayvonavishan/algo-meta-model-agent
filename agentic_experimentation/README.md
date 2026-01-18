Agentic experimentation scaffold for improving the meta model.

Quick start
1) Edit `agentic_experimentation/agent_config.json`.
2) Set your API key env var if you want a hosted LLM (for example `OPENAI_API_KEY`).
3) Refresh the baseline snapshot:
   `python agentic_experimentation/agent_runner.py --config agentic_experimentation/agent_config.json --refresh-baseline`
4) Run one or more iterations:
   `python agentic_experimentation/agent_runner.py --config agentic_experimentation/agent_config.json --iterations 3`

Notes
- Worktrees keep your main repo clean; each iteration runs in its own temp worktree.
- `adaptive_vol_momentum.py` is used as the sweep entry point in the sample config.
- The scoring hook is intentionally minimal; update `scoring_hooks.py` once you decide on the metric.
