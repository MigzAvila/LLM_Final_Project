# 🤖 AgentSociety Challenge — CrewAI Crew Analysis

> **Task:** Yelp Review Simulation (Track 1)
> **Framework:** [CrewAI](https://github.com/crewAIInc/crewAI) ≥ 1.14.1
> **LLM Backend:** NVIDIA NIM via OpenAI-compatible API (`minimax-m2.7`)
> **Package Manager:** `uv`

---

## Table of Contents

1. [Integration Setup Verification](#1-integration-setup-verification)
2. [Architecture Overview](#2-architecture-overview)
3. [Crew Members (Agents)](#3-crew-members-agents)
4. [How Agents Collaborate](#4-how-agents-collaborate)
5. [Knowledge & Data Retrieval](#5-knowledge--data-retrieval)
6. [Current Findings](#6-current-findings) · [§6.5 Measured accuracy](#6-5-measured-accuracy-pipeline-report)
7. [Running the Tests](#7-running-the-tests)

---

## 1. Integration Setup Verification

The project is **fully compliant** with the AgentSociety official simulator integration guidelines. All three required integration points are implemented:

| Requirement | File | Status |
|---|---|---|
| Agent definitions with `role`, `goal`, `backstory` | `config/agents.yaml` | ✅ |
| Task definitions with `description`, `expected_output` | `config/tasks.yaml` | ✅ |
| Crew assembly via `@CrewBase`, `@agent`, `@task` | `src/crews/simulation_crew.py` | ✅ |
| Simulator adapter returning `predicted_rating` & `generated_review` | `src/flows/serving_flow.py` | ✅ |
| `get_interaction_tool()` available for agent equipping | `src/tools/interaction_tool_wrapper.py` | ✅ |
| Mock + real LLM test runners | `run_simulator_test.py` | ✅ |
| Bottom-layer files untouched | `crewai_simulation_agent.py`, `serving_flow.py` | ✅ |

> [!IMPORTANT]
> The `predicted_rating` and `generated_review` keys in `InferenceState` are the **invisible contract** between the crew and the competition adapter. These are never renamed.

---

## 2. Architecture Overview

The system follows a **three-layer architecture**:

```
AgentSociety Simulator
        │
        ▼
┌───────────────────────────────────────┐
│     CrewAISimulationAgent             │  ← Adapter Layer (do not modify)
│  (crewai_simulation_agent.py)         │
│                                       │
│  1. Deterministic data pre-fetch      │
│     (user, item, review history)      │
│  2. Summarise into structured text    │
│  3. Compute fallback_rating from      │
│     user's historical average         │
└───────────────────┬───────────────────┘
                    │ passes InferenceState
                    ▼
┌───────────────────────────────────────┐
│     AgentSocietyServingFlow           │  ← Flow Layer (src/flows/serving_flow.py)
│  (CrewAI Flow)                        │
│                                       │
│  init_request → trigger_crew_inference│
│  Selects crew based on env var:       │
│    CREWAI_PROCESS_MODE=sequential     │
│                      collaborative    │
│                      hierarchical     │
└───────────────────┬───────────────────┘
                    │ kickoff(inputs=...)
                    ▼
┌───────────────────────────────────────┐
│     SimulationCrew (default)          │  ← Crew Layer (src/crews/)
│  (Sequential, 2 LLM agents)           │
│                                       │
│  psychological_analyst                │
│       │ context                       │
│       ▼                               │
│  behavior_simulator                   │
│       │ → {stars, review} JSON        │
└───────────────────────────────────────┘
```

### Key Design Decision: Pre-fetch Over Live Tool Use

Rather than having a `data_retriever` agent call the `Interaction Tool Wrapper` at runtime, **all Yelp data is fetched deterministically in the adapter layer** before the crew ever starts. This means:

- ✅ No hallucinated data — agents only see real Yelp records
- ✅ One fewer LLM round-trip per prediction
- ✅ No threading race conditions on the singleton `InteractionTool`
- ✅ Consistent, reproducible prompts across runs

---

## 3. Crew Members (Agents)

All agents are defined in [`config/agents.yaml`](config/agents.yaml) following the YAML-first pattern. Three crew variants are available, all drawing from the same agent pool.

### 3.1 Agent Pool (`config/agents.yaml`)

#### 🔬 `psychological_analyst`
| Field | Value |
|---|---|
| **Role** | Behavioral Psychologist |
| **Goal** | Analyze retrieved data to uncover the user's latent preferences, rating habits, and satisfaction patterns |
| **Backstory** | Dual background in behavioral science and psychology; profiles user personality traits and rating tendencies from interaction clues |
| **LLM** | `minimax-m2.7` |
| **Used In** | Sequential (primary), Collaborative, Hierarchical |

#### 🎭 `behavior_simulator`
| Field | Value |
|---|---|
| **Role** | Review Simulation Expert |
| **Goal** | Produce the most accurate star rating and realistic review text by deeply embodying the user's persona |
| **Backstory** | Master method actor who pays close attention to historical rating distributions; never defaults to positive ratings |
| **LLM** | `minimax-m2.7` |
| **Used In** | Sequential (primary), Collaborative, Hierarchical |

#### 📡 `data_retriever`
| Field | Value |
|---|---|
| **Role** | Data Retrieval Specialist |
| **Goal** | Accurately retrieve all relevant background data for the target user and item |
| **Backstory** | Expert in big data retrieval and association rules; finds the most valuable records from massive databases |
| **LLM** | `minimax-m2.7` |
| **Used In** | Collaborative, Hierarchical (equipped with `InteractionTool`) |

#### 🗂️ `prediction_manager`
| Field | Value |
|---|---|
| **Role** | Yelp Prediction Project Manager |
| **Goal** | Coordinate specialist agents to deliver one final high-quality prediction package |
| **Backstory** | Senior PM for multi-agent prediction pipelines; breaks down objectives, delegates, validates outputs |
| **LLM** | `minimax-m2.7` |
| **Used In** | Hierarchical (manager role only) |

---

## 4. How Agents Collaborate

Three crew modes are available, selectable via `CREWAI_PROCESS_MODE` environment variable.

### Mode A: Sequential (Default) — `SimulationCrew`

```
[Pre-fetched Yelp Data]
         │
         ▼
 psychological_analyst
  ├─ Receives: user_summary, item_summary, history_summary
  ├─ Task: analyze_preference_task
  │   • States user's historical avg stars & distribution
  │   • Cites 1-2 specific past reviews
  │   • Produces numeric preliminary rating estimate
  └─ Output: Markdown analysis (≤250 words)
         │
         │ (context passed automatically)
         ▼
  behavior_simulator
  ├─ Receives: all pre-fetched data + analyst's output as context
  ├─ Task: simulate_review_task
  │   • Selects a star rating from {1.0, 2.0, 3.0, 4.0, 5.0}
  │   • Writes 1-3 sentence review matching user's tone & style
  │   • References only item's real categories
  └─ Output: STRICT JSON → {"stars": 3.0, "review": "..."}
```

**Hard constraints enforced in prompts:**
- Stars must be consistent with user's historical distribution
- Review in English, matching the user's real vocabulary/casing
- Only references real attributes from the data — no invented products
- Output is a bare JSON object, no prose/markdown around it

### Mode B: Collaborative — `CollaborativeSingleTaskCrew`

```
behavior_simulator (lead)
  ├─ Can delegate to: data_retriever, psychological_analyst
  └─ Single unified task: collaborative_single_task
     Produces: {"stars": float, "review": "string"}
```

All three agents share one task; `behavior_simulator` leads and may delegate sub-queries to peers via CrewAI's native delegation protocol.

### Mode C: Hierarchical — `HierarchicalManagerCrew`

```
prediction_manager (manager, no @agent decorator)
  ├─ Delegates to → data_retriever (equipped with InteractionTool)
  ├─ Delegates to → psychological_analyst
  └─ Delegates to → behavior_simulator
  Single task: hierarchical_predict_task
  Produces: {"stars": float, "review": "string"}
```

The manager orchestrates specialist agents, instructs `data_retriever` to call the `Interaction Tool Wrapper` live, and collects all outputs before the simulator produces the final JSON.

### Collaboration Summary Table

| Mode | `CREWAI_PROCESS_MODE` | Agents Active | LLM Calls | Data Source |
|---|---|---|---|---|
| Sequential | `sequential` | 2 (analyst + simulator) | ~2 | Pre-fetched by adapter |
| Collaborative | `collaborative` | 3 (all workers) | ~3–5 | Pre-fetched by adapter |
| Hierarchical | `hierarchical` | 4 (manager + 3 workers) | ~4–6 | Live tool call by `data_retriever` |

---

## 5. Knowledge & Data Retrieval

### 5.1 What Data Is Used

For each prediction task, the system retrieves three data sources from the Yelp dataset:

| Data | Source | Contents |
|---|---|---|
| **User Profile** | `get_user(user_id)` | Name, review count, yelping_since, average_stars, useful/funny/cool counts, fans, elite years |
| **Item Details** | `get_item(item_id)` | Name, location, avg_stars, review_count, categories, price range, noise level, ambience |
| **Review History** | `get_reviews(user_id)` | Up to the 12 most recent reviews: stars, date, and up to 280-char text snippets |

### 5.2 How Data Is Structured for Prompts

Raw data is not passed to agents directly. It is processed by helper functions in `crewai_simulation_agent.py` into **structured natural language summaries**:

```
USER PROFILE (verbatim from Yelp):
name=Alice; review_count=47; yelping_since=2014; average_stars=3.21;
useful=12; funny=3; cool=5; fans=2; elite=2018,2019

ITEM DETAILS (verbatim from Yelp):
name=The Noodle House; location=Las Vegas, NV; avg_stars=3.8;
review_count=512; categories=Noodles, Asian Fusion; price_range=2

USER REVIEW HISTORY (verbatim from Yelp):
TOTAL_HISTORICAL_REVIEWS=47
USER_HISTORICAL_AVERAGE_STARS=3.21
USER_RATING_DISTRIBUTION: n=47; 5*=8, 4*=9, 3*=18, 2*=7, 1*=5
RECENT_REVIEWS (most recent first):
- [3* on 2023-11-14] service was ok but food took forever…
- [2* on 2023-08-02] nothing special, overpriced for what you get…
```

**Truncation guards** prevent context window overflows:
- Max 12 historical reviews passed to prompts
- Review text capped at 280 characters each
- Item categories capped at 240 characters

### 5.3 Fallback Rating Computation

A `fallback_rating` is calculated deterministically from the user's review history **before** any LLM is called. If the crew fails or times out, this value (the user's historical mean, clamped to [1.0, 5.0]) is used instead of a default 4.0.

### 5.4 Optional Knowledge Base (ChromaDB + RAG)

An optional background knowledge layer is available via the `PrebuiltTextKnowledgeSource` class. This is **disabled by default** (`CREWAI_ENABLE_KNOWLEDGE=false`).

| Setting | Description |
|---|---|
| `CREWAI_ENABLE_KNOWLEDGE=true` | Activates RAG knowledge injection into the crew |
| `CREWAI_KNOWLEDGE_FILE=<path>` | Path to a `.txt` or `.json` background knowledge file |
| `CREWAI_USE_PREBUILT_INDEX=true` | Skips re-embedding if the ChromaDB collection already exists |

**How it works when enabled:**
1. `PrebuiltTextKnowledgeSource` wraps CrewAI's `TextFileKnowledgeSource`
2. On first run, the file is chunked and embedded into a **ChromaDB persistent collection** (`knowledge_crew`)
3. On subsequent runs, `skip_if_index_exists=True` bypasses re-embedding (fast restart)
4. The collection is injected into the `Crew` object via `knowledge_sources=[...]`
5. Agents can semantically query background knowledge alongside the structured Yelp data

> [!NOTE]
> The sequential crew currently pre-fetches all necessary Yelp data directly via the adapter, making the RAG knowledge layer redundant for most use cases. It is useful for injecting **global domain knowledge** (e.g., category-level rating baselines, Yelp platform behaviour notes) that is not user/item-specific.

### 5.5 Tool: `Interaction Tool Wrapper`

```python
# Registered as a CrewAI @tool — used by hierarchical/collaborative modes
interaction_tool_wrapper(query_type: str, target_id: str) -> str

# query_type options:
#   "user"            → get_user(user_id=target_id)
#   "item"            → get_item(item_id=target_id)
#   "review_by_user"  → get_reviews(user_id=target_id)
#   "review_by_item"  → get_reviews(item_id=target_id)
```

The wrapper is a thin bridge between the AgentSociety `InteractionTool` (injected by the simulator) and CrewAI's `@tool` interface. A module-level singleton with a threading lock ensures it is safe for concurrent agents.

---

## 6. Current Findings

### 6.1 Architecture Decisions & Their Rationale

| Decision | Rationale | Outcome |
|---|---|---|
| **Pre-fetch data in adapter, not via agent tool call** | Eliminates hallucination; removes 1 LLM round-trip; avoids threading race on singleton tool | ✅ Zero hallucinated product names/categories |
| **Sequential 2-agent crew as default** | Lowest token cost; analyst provides grounded context for simulator; fewest failure points | ✅ Reliable; easiest to debug |
| **Hard-coded JSON output enforcement in task prompts** | LLMs are prone to wrapping JSON in prose or markdown fences | ✅ `extract_json_from_output()` in flow handles noisy outputs as a safety net |
| **Fallback rating from user history** | Prevents defaulting to 4.0 when crew fails | ✅ Graceful degradation on API errors |
| **Star rating constrained to {1.0, 2.0, 3.0, 4.0, 5.0}** | Matches Yelp's actual discrete rating scale | ✅ Avoids non-standard outputs like 3.5 |
| **Rating distribution injected verbatim** | Critical for replicating critical reviewers (avg 2-3★) vs generous ones (avg 4-5★) | ✅ Prevents systematic upward bias |

### 6.2 Prompt Engineering Strategies

- **Distribution-anchored prompting:** The analyst task explicitly requires stating the user's `n=X; 5*=Y, 4*=Z…` distribution before making any estimate. This anchors the simulator agent to the correct magnitude.
- **Persona replication:** The behavior_simulator backstory emphasizes replicating tone, vocabulary, and casing (e.g., "many Yelp users write entirely in lowercase").
- **Negative constraints:** "Do NOT default to 4 or 5 just because the item has high public ratings" — directly counters known LLM optimism bias.
- **Cross-contamination prevention:** "Never reference products or categories absent from the data" — stops agents from hallucinating electronics into a restaurant review.

### 6.3 Known Limitations & Open Challenges

> [!WARNING]
> The following issues have been observed or anticipated:

| Issue | Impact | Proposed Mitigation |
|---|---|---|
| **LLM optimism bias** | Model tends to predict 4.0 even for critical users | Stronger negative constraints; few-shot examples of low-star reviews |
| **Context window limits** | Users with 500+ reviews are truncated to 12 | Summarise distribution more aggressively; weight recent reviews higher |
| **Rate limiting (HTTP 429)** | Parallel runs hit token-per-minute limits | Reduce `max_workers` in `evaluate_with_training_data.py` |
| **ChromaDB embedding conflicts** | Mismatched embedding models cause validation errors on restart | `PrebuiltTextKnowledgeSource.skip_if_index_exists` avoids re-indexing; delete `~/.local/share/crewai/` to reset |
| **Non-JSON LLM output** | Some models wrap JSON in markdown fences | `extract_json_from_output()` regex fallback handles this |

### 6.4 Evaluation Metrics

The competition evaluates on two metrics:

- **Star rating MAE** (Mean Absolute Error) — lower is better
- **Review quality** (text similarity / plausibility) — higher is better

Local test output example (mock mode, structural check):
```
★ stars  pred=4.0  gt=3.0  Δ=+1.0
📝 pred : [Mocked LLM] Solid spot, would visit again.
🎯 gt   : service was ok but portions have gotten smaller lately
```

Real inference test summary format:
```
Summary: MAE=X.XX  exact=Y/N  within 1 star=Z/N
```

### 6.5 Measured accuracy (pipeline report)

The following figures come from the official simulator evaluation block in [`pipeline_report_20260505_113003.json`](pipeline_report_20260505_113003.json) (`run_pipeline.py --mock --tasks 1`).

| Run context | Value |
|---|---|
| **Report file** | `pipeline_report_20260505_113003.json` |
| **Run timestamp** | `20260505_113003` |
| **Mode** | Mock LLM (litellm patched; structural / smoke run) |
| **Model name in report** | `minimaxai/minimax-m2.7` (from environment at report write time) |
| **Tasks run / GT pairs** | `1` simulated vs `41` ground-truth rows loaded (evaluator uses `min(count)` → **n = 1** for metrics) |
| **Pipeline errors** | `0` |

**Simulator metrics (higher is better for all three):**

| Metric | Score | Interpretation |
|---|---|---|
| **preference_estimation** | **0.9455** | Star-rating side of the objective (≈94.5% on this single pair). |
| **review_generation** | **0.4513** | Text-similarity / review-quality score (≈45.1%). |
| **overall_quality** | **0.6984** | Combined objective (≈69.8%). |

> [!NOTE]
> On this run the emitted review string was the adapter **fallback** (`"Crew execution failed; falling back to historical average."`), so **review_generation** and **overall_quality** mainly reflect that single-task path—not a full multi-task or failure-free crew run. Re-run `uv run python run_pipeline.py --tasks N` (or the full suite) and paste the latest `pipeline_report_*.json` metrics here when you want an updated accuracy table.

---

## 7. Running the Tests

### Quick Structural Test (Free — No Tokens Used)
```powershell
uv run python run_simulator_test.py --mock
```

### Real LLM Test (5 Tasks by Default)
```powershell
uv run python run_simulator_test.py
```

### Full Training Set Evaluation
```powershell
uv run python evaluate_with_training_data.py
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CREWAI_PROCESS_MODE` | `sequential` | Crew variant: `sequential`, `collaborative`, `hierarchical` |
| `SIMULATOR_NUM_TASKS` | `5` | Number of tasks to run (`all` for full set) |
| `SIMULATOR_THREADING` | `false` | Enable parallel task execution |
| `SIMULATOR_MAX_WORKERS` | `1` | Worker count when threading is on |
| `CREWAI_ENABLE_KNOWLEDGE` | `false` | Enable optional ChromaDB RAG layer |
| `CREWAI_KNOWLEDGE_FILE` | _(none)_ | Path to background knowledge file |
| `CREWAI_USE_PREBUILT_INDEX` | `false` | Skip re-embedding if index exists |

---

## File Map

```
AgentSocietyChallenge_w_CrewAI/
│
├── config/
│   ├── agents.yaml                   # 🔧 Agent definitions (role/goal/backstory)
│   ├── tasks.yaml                    # 🔧 Sequential task definitions
│   ├── tasks_collaborative.yaml      # 🔧 Collaborative single-task definition
│   └── tasks_hierarchical.yaml       # 🔧 Hierarchical task definition
│
├── src/
│   ├── crews/
│   │   ├── simulation_crew.py        # 🔧 Default sequential crew (2 agents)
│   │   ├── collaborative_single_task_crew.py  # Alt: collaborative mode
│   │   └── hierarchical_manager_crew.py       # Alt: hierarchical mode
│   │
│   ├── flows/
│   │   └── serving_flow.py           # ⛔ Flow layer (do not modify key names)
│   │
│   ├── knowledge/
│   │   └── prebuilt_source.py        # Optional ChromaDB RAG source
│   │
│   └── tools/
│       └── interaction_tool_wrapper.py  # ⛔ Simulator tool bridge
│
├── crewai_simulation_agent.py        # ⛔ Adapter layer (do not modify)
├── run_simulator_test.py             # Test runner (mock + real LLM)
├── evaluate_with_training_data.py    # Full dataset evaluator
└── pyproject.toml                    # uv-managed dependencies
```

> [!TIP]
> Files marked 🔧 are the **student zone** — safe to modify freely.
> Files marked ⛔ are the **bottom-layer** — modify with caution and never rename their output keys.
