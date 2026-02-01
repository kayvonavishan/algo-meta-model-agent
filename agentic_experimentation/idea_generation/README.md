# Idea generation (Claude Agent SDK)

This folder contains a small script to generate new meta-model improvement ideas by calling the Claude Agent SDK once per idea, injecting:

- `agentic_experimentation/prompts/idea_generator/idea_generator_prompt.txt`
- `META_MODEL_GUIDE.md`
- `adaptive_vol_momentum.py`
- prior ideas in `agentic_experimentation/ideas/` and `agentic_experimentation/ideas/completed/`

## Usage

```powershell
python agentic_experimentation/idea_generation/generate_ideas.py --count 3
```

Optional:

```powershell
python agentic_experimentation/idea_generation/generate_ideas.py --count 1 --model claude-3-7-sonnet
```

## Requirements

- `claude-agent-sdk` installed (it bundles the Claude Code runtime).
- `ANTHROPIC_API_KEY` set (this repo's runners load `agentic_experimentation/.env` automatically; this script also loads it).
