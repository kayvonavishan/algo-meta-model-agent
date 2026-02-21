# Idea Ledger

The idea ledger is a persistent, append-only history of ideas evaluated by `tree_runner.py`.
It lives outside ephemeral worktrees so you can:

- Track all ideas that were tested
- Preserve order and lineage (path through the tree)
- Segment history by "area" of the codebase
- Feed prior ideas into new runs as context

## Where It Lives

Default root:
```
agentic_experimentation/idea_ledger/
```

Per-area layout:
```
agentic_experimentation/idea_ledger/<area_id>/
  runs/<run_id>/node_ideas/<node_id>/*.md
  context/*.md
  index.jsonl
```

- `runs/` stores archived idea files (never deleted by cleanup).
- `context/` is a rolling window of recent ideas used as generator context.
- `index.jsonl` is the append-only event log of idea usage and outcomes.

## How It Works

The ledger is updated automatically by `tree_runner.py`:

- When an idea is queued for evaluation, a `queued` event is recorded.
- When an evaluation completes or fails, a `completed` or `failed` event is recorded.
- Idea files are copied into `runs/<run_id>/node_ideas/<node_id>/`.
- A copy of the idea is also placed in `context/` (latest N, configurable).

Each ledger record captures:

- `area_id`, `tree_run_id`, `eval_id`, `node_id`, `parent_node_id`, `depth`
- `idea_hash`, `idea_text`, `idea_path`, `idea_archive_path`
- `idea_chain` and `path_node_ids` (ordered lineage)
- `root_commit`, `candidate_commit`
- `status` and `decision`

## Using Area IDs

Use `--area-id` to segment idea history per domain of your codebase:

```
python agentic_experimentation/tree_runner.py --area-id meta_model
python agentic_experimentation/tree_runner.py --area-id risk_model
```

Each area gets its own ledger subdirectory and index.

## Feeding Ledger Context into Idea Generation

The idea generator uses context directories. By default, context includes:

- node-plus-ancestors ideas (from the current tree)
- the ledger `context/` directory (recent ideas from prior runs)

You can control this with:

```
--ideas-context-strategy node_plus_ancestors+ledger
--idea-ledger-context-limit 200
```

## CLI Flags

- `--area-id` (default: `default`)
- `--idea-ledger-root` (default: `agentic_experimentation/idea_ledger`)
- `--idea-ledger-context-limit` (default: `200`)
- `--ideas-context-strategy` (default: `node_plus_ancestors+ledger`)

## Cleanup Guidance

You can safely delete `agentic_experimentation/worktrees/` and
`agentic_experimentation/experiments/` between runs. The idea ledger
remains intact for long-term history and future context.

