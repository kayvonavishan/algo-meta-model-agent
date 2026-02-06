# Idea Conversation Continuation — Step-by-Step Implementation Plan

## Objective
Add branch-aware idea-generation conversation continuation so each node in tree search has its own conversation lineage, with child nodes forking from a parent checkpoint.

---

## Phase 0 — Lock behavior + defaults
1. Confirm continuation semantics:
   - One conversation per node branch.
   - Child nodes fork from parent checkpoint (`A -> B/C/D`).
2. Add config defaults in `tree_runner.py` CLI:
   - `--idea-conversation-mode` (`off|auto|native|replay`, default `auto`)
   - `--idea-history-window-turns` (default e.g. `12`)
   - `--idea-history-max-chars` (default e.g. `20000`)
3. Define fallback policy:
   - `native` if provider continuation IDs are available.
   - `replay` otherwise.

**Acceptance criteria**
- New CLI flags parse and appear in run config.
- Default behavior unchanged when mode is `off`.

---

## Phase 1 — Manifest schema additions
1. Bump `manifest_version` in `tree_runner.py`.
2. Add top-level `conversation_config`:
   - `mode`, `history_window_turns`, `history_max_chars`.
3. Add top-level `conversations` map keyed by `conversation_id`.
4. Add node fields:
   - `conversation_id`
   - `expansion_seed_turn_id`
   - `latest_conversation_turn_id`
5. Add evaluation fields:
   - `idea_generation_conversation_id`
   - `idea_generation_turn_id`
6. Add manifest migration helper for resume on older manifests.

**Acceptance criteria**
- New manifests include conversation sections.
- Old manifests resume without crashing and are upgraded in-memory/on-write.

---

## Phase 2 — Conversation artifact storage
1. Create run-local directory:
   - `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/conversations/`
2. Add file conventions:
   - `<conversation_id>.json` (state)
   - `<conversation_id>.jsonl` (turn log)
   - `<conversation_id>_summary.md` (compressed memory)
3. Add utility functions in `tree_runner.py` (or small helper module):
   - create conversation
   - append turn
   - write/read summary
   - checkpoint a turn id

**Acceptance criteria**
- Files are created atomically.
- Retries do not duplicate identical turns (idempotency key).

---

## Phase 3 — `generate_ideas.py` conversation I/O
1. Add args:
   - `--conversation-state-in`
   - `--conversation-state-out`
   - `--conversation-mode`
   - `--fork-from-turn-id` (optional)
2. Load conversation state on startup.
3. For each generation call:
   - `native`: continue with provider ID/session if available.
   - `replay`: prepend summary + bounded recent turns.
4. Capture and persist:
   - provider session/response ids (if available)
   - prompt hash
   - turn id
   - timestamp
   - output idea file path
5. Write updated conversation state to `--conversation-state-out`.

**Acceptance criteria**
- `generate_ideas.py` can run with/without conversation files.
- In `replay` mode, prompt includes bounded branch memory.

---

## Phase 4 — Wire orchestration in `tree_runner.py`
1. Root setup:
   - create root conversation and attach to node `0000`.
2. Before node expansion:
   - pass node conversation state paths to `generate_ideas.py`.
3. After idea generation:
   - store parent expansion checkpoint as `expansion_seed_turn_id`.
4. On promotion:
   - create child conversation from parent checkpoint.
   - set `parent_conversation_id`, `fork_from_turn_id`.
5. Record eval linkage:
   - save `idea_generation_conversation_id` and `idea_generation_turn_id`.

**Acceptance criteria**
- Beam width > 1 creates sibling child conversations forked from same parent checkpoint.
- Grandchildren continue from their own branch conversation, not siblings.

---

## Phase 5 — Memory compaction + token safety
1. Add deterministic memory packer:
   - latest `N` turns + compact summary.
2. Enforce char budget from config.
3. If overflow:
   - trim oldest turns first.
   - keep summary + newest turns.
4. Keep existing artifact/status sections in prompt unchanged.

**Acceptance criteria**
- Prompt size remains within limits.
- Branch continuity context persists across many depths.

---

## Phase 6 — Resume/retry/idempotency hardening
1. On resume:
   - rebuild in-memory conversation pointers from manifest + files.
2. Retry safety:
   - no duplicate turn records for same operation.
3. Stale/partial files:
   - detect and recover with clear warnings.

**Acceptance criteria**
- Interrupted run resumes without losing branch conversation lineage.
- Repeat execution of same step does not append duplicate turns.

---

## Phase 7 — Reporting + visibility
1. Extend `TREE_SUMMARY.md`:
   - per-node `conversation_id`, `parent_conversation_id`, `latest_turn_id`.
2. Extend `TREE_GRAPH.md`:
   - show conversation fork lineage near node edges.
3. Add optional conversation debug log path in manifest.

**Acceptance criteria**
- Users can trace both model lineage and conversation lineage per node.

---

## Phase 8 — Tests
1. Unit tests:
   - manifest migration
   - conversation create/fork/update
   - memory packing and truncation
   - idempotent append behavior
2. Dry-run integration test:
   - `beam_width=3`, `max_depth=2`
   - assert lineage:
     - root conv A
     - children from A checkpoint
     - grandchildren continue B/C/D independently
3. Resume test:
   - interrupt mid-depth and verify continued conversations.

**Acceptance criteria**
- Test suite passes for new coverage.
- No regressions in existing tree runner tests.

---

## Phase 9 — Documentation
1. Update `agentic_experimentation/TREE_RUNNER_GUIDE.md`:
   - new flags
   - branch conversation model
   - where conversation files live
   - troubleshooting native vs replay mode
2. Add one example run command for each mode:
   - `off`, `auto`, `replay`.

**Acceptance criteria**
- A user can run and debug conversation continuation from docs only.

---

## Recommended execution order (minimal risk)
1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 6
6. Phase 5
7. Phase 7
8. Phase 8
9. Phase 9

---

## Files expected to change
- `agentic_experimentation/tree_runner.py`
- `agentic_experimentation/idea_generation/generate_ideas.py`
- `agentic_experimentation/TREE_RUNNER_GUIDE.md`
- `tests/*` (new/updated tree runner + idea generation tests)

