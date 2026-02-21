# Example Idea Generation Prompt (Initial / Root Node)

This example shows the prompt shape for the **very first idea generation call** in a new tree run (root node, depth 0).

Notes:
- Values/paths are illustrative.
- This is for node `0000` before any idea has been promoted.
- For a first call with no prior turns, the replay-memory block is absent.
- A "New Ideas Already Introduced" section is still appended at the end.

---

You are a quantitative trading researcher developing a new trading model. We have a baseline model and performance metrics (sharpe, average profit pertrade etc) for the baseline state. Our current approach is to focus on a single area of the model and make small interative changes, testing performance after each change, and selecting canidates to move forward with based on increases in performance.

Code Area of Focus:
- We will be focusing on the "meta model" area of our existing quantitative trading strategy.
- The meta model selects top-performing base strategies for the next period using training-free signals (see adaptive_vol_momentum.py).
- You will be given additional read-only repo context below (META_MODEL_GUIDE.md, adaptive_vol_momentum.py, scoring.py, selection.py).
- You will also be given branch timeline and rejected-idea summaries (idea file paths + metrics) so you can avoid duplicating ideas already tested.

Experiment Description:

- We are iteratively improving the meta model using a beam-search style process: generate multiple candidate ideas, implement/test them, then promote the most promising changes as new "nodes" in an improvement tree/branch.
- Below you will find information about the current status of this experiment (where we are in the branch), which ideas have been applied, and how performance has changed.
- You may also be given a "BASELINE ARTIFACTS (READ-ONLY, OPTIONAL)" section with baseline metrics + file paths to sweep artifacts; use it to inspect results as needed, but do not load everything unnecessarily.

Testing Strategy:

- After implementing each model enhancement idea, we test the performance of the updated model comprehensively.
- The current meta model includes a set of input params, each of which can be set to a range of numbers.
- We currently have a pre-generated set of 25 param sets.
- We run the model on all 25 and generate performance metrics for all 25 param sets.
- We also aggregate metrics such as averages across all 25 model backtests.
- We call this our "sweep".

Output Guidelines:

- Ideas must be implementable as a single coding change (idea -> implement).
- No multi-step experiments inside one idea. If you think multiple variants could help, emit them as separate ideas, not "try A then B".
- Each idea must include all context needed for a quant developer to implement it, even if that means repeating information across ideas.
- Provide specific details on what needs to change in the context of the current meta model implementation.
- Exact code changes are not necessary, but helpful toy examples are fine.
- Pick the idea you think could be most positively impactful.

Return exactly:
IDEA: <one concise, self-contained change>
RATIONALE: <why it might help, in plain terms>
REQUIRED_CHANGES: <elaborate on changes required to the meta model. Detailed code changes are not necessary>

-----------------------

Current Status of Beam Search Experiment:
- tree_run_id=20260206 node_id=0000 depth=0 changes_applied_to_original=0
- applied_ideas: (none; original/root model)

Artifacts & How To Interpret Them:
- Below are locations and descriptions model performance artifacts. These are generated from the most recent form of the model and represent performance characteristics of the current state.
- These files can be used to understand the current behavior, strengths, weaknesses and overall performance of the model. This can help you strategize where you may be able to improve the model.
- Several output artifacts are csv files. To aid we have provided .txt files with column definitions. They can be found in  `agentic_experimentation/artifact_docs/`

Output: meta_config_sweep_results.csv
- What it stores: Per-config sweep results; each row is one meta-model backtest for a single parameter set (`config_id`).
- Used for: Comparing parameter sets and computing the averaged metrics/deltas used to judge/promote ideas.
- Location (current node): agentic_experimentation/baselines/meta_config_sweep_results_baseline.csv
- Exists (current node path): true
- Column definitions: c:\Users\micha\myhome\git\algo-meta-model-agent\agentic_experimentation\artifact_docs\meta_config_sweep_results_columns.txt (exists=true)
  - Format: `column_name: description` (search by column name).
  - Note: metrics families include `core_*`, `rel_*`, `stab_*`, `trade_*`, `sig_*`.
- Granularity: 1 CSV per node/run; rows are per parameter set tested (`config_id`).

Output: avg_trade_return_plots/
- Availability: not available yet for this node.
- Why: these diagnostics are produced only after running the first candidate idea sweep.
- Location (current node): (not generated yet)
- Overview doc: c:\Users\micha\myhome\git\algo-meta-model-agent\agentic_experimentation\artifact_docs\avg_trade_return_plots\README.txt (exists=true)
- Note: once available, files are per `config_id` with suffixes `_config_000`, `_config_001`, ...


Branch Timeline (chronological):
0. node_id=0000 depth=0
   applied_idea: (root/original)
   sweep_config_limit: 25 (uses config_id < 25)
   core_topN_sharpe: 1.6
   mean_topN_avg_return_per_trade_pct_oos: 0.03
   mean_topN_avg_return_per_trade_pct: 0.135
   core_topN_sortino: 3.7
   core_topN_calmar: 2.5
   core_topN_max_drawdown: -0.25
   baseline_results_csv_path: agentic_experimentation/baselines/meta_config_sweep_results_baseline.csv

===== META MODEL CONTEXT =====
Below is a list of files to explore to learn and understand the current state of the meta model. You can explore additional files, such as imports, but the files below make up the core modules.
META_MODEL_GUIDE.md
 - description: High-level guide for the meta model design, assumptions, and workflow.
 - location: c:\Users\micha\myhome\git\algo-meta-model-agent\META_MODEL_GUIDE.md
adaptive_vol_momentum.py
 - description: Primary meta model implementation and sweep/backtest driver.
 - location: c:\Users\micha\myhome\git\algo-meta-model-agent\adaptive_vol_momentum.py
scoring.py
 - description: Performance scoring and summary metric computation utilities.
 - location: c:\Users\micha\myhome\git\algo-meta-model-agent\scoring.py
selection.py
 - description: Selection logic used to choose top strategies/models each period.
 - location: c:\Users\micha\myhome\git\algo-meta-model-agent\selection.py

NOTE: Review the 'New Ideas Already Introduced' section at the end and do not repeat those ideas.

===== New Ideas Already Introduced =====
- (none yet)
