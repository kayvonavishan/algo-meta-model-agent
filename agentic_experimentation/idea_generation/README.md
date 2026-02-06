# Idea generation (Claude Agent SDK)

This folder contains a small script to generate new meta-model improvement ideas by calling the Claude Agent SDK once per idea, injecting:

- `agentic_experimentation/prompts/idea_generator/idea_generator_prompt.txt`
- `META_MODEL_GUIDE.md`
- `adaptive_vol_momentum.py`
- prior ideas in `agentic_experimentation/ideas/` (including not-yet-tested ideas) and `agentic_experimentation/ideas/completed/`

## Usage

```powershell
python agentic_experimentation/idea_generation/generate_ideas.py
```

Edit `agentic_experimentation/idea_generation/config.json` to change `count`, `model`, and `max_context_chars`.

If you see Windows errors like `[WinError 206] The filename or extension is too long`, set `cli_path` in the config to your system `claude` executable path (or leave it as `null` and the script will try to auto-detect `claude` on PATH).

Note: the script uses Agent SDK "streaming mode" so large prompts are sent over stdin (avoids Windows command-line length limits).

If runs still fail with "Command failed with exit code 1", check:
- standalone usage: `agentic_experimentation/logs/idea_generation/claude_debug/`
- tree runner usage: `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/logs/idea_generation/claude_debug/`

## Requirements

- `claude-agent-sdk` installed (it bundles the Claude Code runtime).
- `ANTHROPIC_API_KEY` set (this repo's runners load `agentic_experimentation/.env` automatically; this script also loads it).
