# SkillGraph Pipeline Redesign

This note proposes a minimal but meaningful upgrade of the current `SkillGraph` pipeline.

The goal is not to make the system much more complicated. The goal is to make the graph representation support the actual research story:

- automatic skill distillation from strong-model trajectories,
- graph-based organization instead of flat skill files,
- and inference-time guidance for weaker SWE agents in dynamic environments.

## 1. The right story

The project should not be framed as:

> "We store distilled skills in a graph instead of markdown files."

That is too weak.

The stronger framing is:

> Existing memory/skill methods mostly produce static knowledge artifacts. But SWE agents operate in dynamic environments where the next useful decision depends on intermediate observations, failures, and tool outcomes. We therefore need a structured skill library that can guide branching and recovery at inference time.

This gives you three concrete claims:

1. Skills are not only reusable content; they are reusable decision units.
2. The structure between skills matters, not just the content of each skill.
3. Graphs are useful because they provide execution-time routing for smaller models.

## 2. What to keep from the current pipeline

Keep the current two-stage structure:

1. `intra-task`: build one task-specific graph from multiple trajectories.
2. `cross-task`: merge task graphs into a global graph.

This is already a strong design choice. Do not throw it away.

Also keep:

- successful trajectories as the main source of reusable strategies,
- failed trajectories as the source of pitfalls and alternative routing,
- graph nodes as generalized decision patterns rather than raw actions.

## 3. What is missing in the current pipeline

The current implementation already extracts nodes and edges, but the graph is still closer to a static knowledge graph than a guidance graph.

The main gaps are:

1. State is implicit.
   - `triggers` and edge `condition` are plain text, but there is no explicit notion of input state, output state, or failure state.

2. Edge semantics are too weak.
   - `reveals` and `if_fails` are a good start, but not enough for guidance.

3. There is no inference-time router.
   - The pipeline builds a graph, but there is no mechanism that lets an agent select entry nodes and traverse the graph online.

4. Merge uses weak evidence.
   - It mostly uses description/triggers text similarity plus an LLM judge, but not node role, predecessor/successor structure, or observed outcomes.

5. Reliability is not represented.
   - Nodes and edges do not yet carry support counts, source tasks, or confidence.

## 4. Minimal redesign: turn the graph into a state-aware guidance graph

Do not introduce a heavy symbolic planner. Keep the pipeline LLM-friendly and simple.

### 4.1 Upgrade node schema

Each node should describe a reusable decision unit with explicit state interface:

- `description`: one-line summary
- `preconditions`: when this node should be considered
- `observation_signals`: what evidence typically activates it
- `goal`: what local problem this node is trying to solve
- `actions`: what to do
- `success_signals`: what observations indicate the node worked
- `failure_signals`: what observations indicate the node did not work
- `verification`: how to validate the local outcome
- `pitfalls`: common mistakes
- `support`: number of trajectories supporting this node
- `provenance`: task ids / trajectory ids

Important simplification:

- `preconditions` and `observation_signals` can be short natural-language fields.
- They do not need to be fully formalized.
- The router can still use LLM matching over these fields.

### 4.2 Upgrade edge schema

Replace the current loose edge design with a slightly richer but still simple schema:

- `type`: one of
  - `next`
  - `branch_on_observation`
  - `branch_on_failure`
  - `branch_on_verification`
  - `alternative`
- `condition`: the observation that activates the edge
- `rationale`: optional short explanation
- `support`: how many trajectories showed this transition

This is enough to tell a guidance story without introducing complicated program logic.

### 4.3 Add a small runtime state abstraction

At inference time, define a compact state snapshot:

- current subgoal,
- last action type,
- key recent observation,
- whether the last attempt succeeded / failed / is ambiguous,
- current tool context (search / read / edit / test / verify).

This can be produced by a small LLM summarizer from the agent's recent context.

## 5. Revised pipeline

## Stage A. Parse trajectories into state transitions

Current parser output:

- thought
- action
- action type
- observation

Add one more lightweight pass:

- summarize each step into:
  - `state_before`
  - `decision`
  - `state_after`

Do not force a formal schema for every token in the trajectory. A short natural-language summary per step is enough.

Why this matters:

- the graph should be induced from decision transitions, not just from action spans.

## Stage B. Intra-task graph induction

The current `initializer -> success updater -> failure updater` structure is good. Keep it.

But change what each component produces.

### Initializer

Instead of only extracting nodes plus `reveals / if_fails`, ask for:

- nodes with `preconditions`, `success_signals`, `failure_signals`,
- edges that describe what observation causes the next node to become relevant.

### Success updater

Current behavior:

- update node
- add node
- split node
- add edge

Keep this API. Add two stricter rules:

1. Prefer `update_node` when the local goal is the same but signals/actions become more precise.
2. Use `split_node` when one node actually mixes two different state interfaces.

This is the most important criterion for graph quality:

- nodes should be separated by different activation conditions or different local goals,
- not merely by different wording.

### Failure updater

Current behavior only adds pitfalls.

Upgrade it slightly:

- still add pitfalls,
- but also allow adding `branch_on_failure` edges when a failed trajectory shows a reliable recovery path.

This is critical for your story because failed trajectories are exactly where dynamic recovery structure comes from.

## Stage C. Cross-task graph induction

Keep the current merge stage, but strengthen the matching signal.

When matching nodes across tasks, use:

1. `description`
2. `preconditions`
3. `goal`
4. `success_signals`
5. local topology:
   - predecessor types
   - successor types

Minimal implementation:

- expand the embedding text to include these fields,
- include a node's incoming/outgoing edge summaries in the LLM judge prompt.

This is much cheaper than designing a fully new graph matching algorithm.

### Add support-aware merge

After merge, each unified node should accumulate:

- support count,
- source task ids,
- source trajectory ids.

This helps both research claims and runtime use:

- higher-support nodes can be preferred by the router,
- provenance makes the graph inspectable.

## Stage D. Post-merge cleanup

Add one simple cleanup pass after cross-task merge:

1. prune nodes with very low support and weak transferability,
2. merge duplicate edges,
3. limit node out-degree by keeping the best-supported or most distinct transitions.

This keeps the graph navigable and avoids a dense, noisy library.

## Stage E. Inference-time router

This is the missing piece in the current codebase.

The router can be very simple:

1. Summarize the current runtime state.
2. Retrieve top-k candidate nodes by embedding.
3. Let an LLM choose:
   - the best entry node,
   - whether to follow an outgoing edge from the current node,
   - or whether to backtrack to an alternative node.
4. Load only the selected node and a 1-hop neighborhood.

This is enough to support the central claim:

- the graph provides execution-time guidance,
- and the agent does not need to read the whole skill library.

## 6. Concrete code changes

These are the lowest-risk upgrades to the current codebase.

### `skillgraph/models.py`

Add structured metadata rather than storing only markdown:

- keep markdown for persistence and readability,
- but also keep parsed node metadata and edge metadata,
- or add helper functions to parse frontmatter into fields.

At minimum, store:

- support,
- provenance,
- edge support.

### `skillgraph/graph/prompts.py`

Revise prompts so node JSON includes:

- `preconditions`
- `observation_signals`
- `goal`
- `success_signals`
- `failure_signals`
- `pitfalls` (optional)

Revise edge JSON so it includes the richer edge types.

### `skillgraph/graph/builder.py`

Change:

- initializer prompt output,
- success updater merge logic,
- failure updater so it can optionally add recovery edges,
- support/provenance accumulation.

### `skillgraph/embedding.py`

Change `extract_node_embed_text()` so it embeds:

- description,
- preconditions,
- goal,
- success/failure signals,
- possibly short topology summaries.

### `skillgraph/merge/prompts.py`

Extend the judge prompt to compare:

- activation conditions,
- local goals,
- outcomes,
- local neighborhood.

This should reduce over-merging.

### new module: `skillgraph/runtime/router.py`

Add a small runtime component with three functions:

- `summarize_state(...)`
- `retrieve_entry_nodes(...)`
- `choose_next_node(...)`

This module can initially be offline or used in a simple simulation harness.

## 7. What not to do yet

To keep the project manageable, avoid these for now:

- fully symbolic state machines,
- complex graph neural networks,
- end-to-end RL for graph traversal,
- multi-hop global planning over the full graph,
- very large ontologies of edge types.

These additions may sound impressive, but they are not necessary for a strong first paper/story.

## 8. Recommended evaluation story

Use three baselines:

1. no external skills,
2. flat skill files,
3. skill graph guidance.

Measure:

- task success,
- token/context cost,
- number of nodes read,
- recovery after failed intermediate attempts,
- performance gain on small/open models.

If possible, also report:

- average node support,
- average out-degree,
- retrieval precision of the router,
- how often failure edges are used.

## 9. The simplest strong version of the project

If you want the cleanest and least risky version, the project should claim:

1. frontier trajectories can be distilled into reusable state-aware skill nodes,
2. these nodes can be organized into a graph with conditional transitions,
3. the graph can guide weaker SWE agents by selective retrieval and local traversal,
4. this works better than flat skill documents.

That is already a strong paper. You do not need a more complicated system before you can tell a convincing story.
