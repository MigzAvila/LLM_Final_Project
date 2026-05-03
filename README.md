<div style="text-align: center; display: flex; align-items: center; justify-content: center; background-color: white; padding: 20px; border-radius: 30px;">
  <img src="./static/ASC.jpg" alt="AgentSociety Challenge Logo" width="100" style="margin-right: 20px; border-radius: 10%;">
  <h1 style="color: black; margin: 0; font-size: 2em;">WWW'25 AgentSociety Challenge: WebSocietySimulator</h1>
</div>

# 🚀 AgentSociety Challenge
![License](https://img.shields.io/badge/license-MIT-green) &ensp;
[![Competition Link](https://img.shields.io/badge/competition-link-orange)](https://www.codabench.org/competitions/4574/) &ensp;
[![arXiv](https://img.shields.io/badge/arXiv-2502.18754-b31b1b.svg)](https://arxiv.org/abs/2502.18754)

> **⚠️ DISCLAIMER: Educational Sandbox Template**
> This repository is a **customized educational sandbox version** designed specifically for learning CrewAI integration. It is **NOT** the official Tsinghua University AgentSociety Challenge repository. For the official upstream code, datasets, and competition guidelines, please visit the [official repository](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge).

Welcome to the **WWW'25 AgentSociety Challenge**! This repository provides the tools and framework needed to participate in a competition that focuses on building **LLM Agents** for **user behavior simulation** and **recommendation systems** based on open source datasets.

Participants are tasked with developing intelligent agents that interact with a simulated environment and perform specific tasks in two competition tracks:
1. **User Behavior Simulation Track**: Agents simulate user behavior, including generating reviews and ratings.
2. **Recommendation Track**: Agents generate recommendations based on provided contextual data.

This repository includes:
- The core library `websocietysimulator` for environment simulation.
- Scripts for dataset processing and analysis.
- Example usage for creating and evaluating agents.

---

## How it works

1. **`Simulator`** (`websocietysimulator`) loads your dataset (`yelp_dataset/`), task definitions (`yelp_tasks/`), and optional ground truth (`yelp_groundtruth/`). For each task it invokes your agent with the active user–item scenario.

2. **`CrewAISimulationAgent`** (`crewai_simulation_agent.py`) is the bridge class: it uses the simulator’s `InteractionTool` to fetch user, item, and review history, turns them into compact text summaries, and passes them into a CrewAI **Flow**.

3. **`AgentSocietyServingFlow`** (`src/flows/serving_flow.py`) reads `CREWAI_PROCESS_MODE` and runs one of three crews:
   - **sequential** → `SimulationCrew`
   - **collaborative** → `CollaborativeSingleTaskCrew`
   - **hierarchical** → `HierarchicalManagerCrew`

4. The crew produces a **predicted star rating** and **generated review** text. The flow parses JSON (or falls back to regex), clamps stars to 1–5, and returns them to the simulator.

5. **`Simulator.evaluate()`** compares predictions to ground truth (when present) and reports metrics (for example RMSE-style error and sentiment-related scores), depending on the competition tooling.

Optional **knowledge / RAG**: when `CREWAI_ENABLE_KNOWLEDGE=true`, `SimulationCrew` can attach a Chroma-backed knowledge source built from a JSON file (see [Optional: knowledge index](#optional-prebuild-and-reuse-crewai-knowledge-index)). The default path uses structured Yelp rows already injected into prompts, so RAG is off unless you enable it.

---

## Workflow

Typical end-to-end flow:

```mermaid
flowchart LR
  subgraph setup [Setup]
    A["uv sync"] --> B["Copy .env.example → .env"]
    B --> C["Place Yelp-format data under yelp_dataset/"]
  end
  subgraph run [Evaluate]
    D["run_simulator_test.py"] --> E["Simulator + CrewAISimulationAgent"]
    E --> F["AgentSocietyServingFlow"]
    F --> G["Crew crew.kickoff"]
    G --> H["stars + review"]
    H --> I["evaluate vs ground truth"]
  end
  setup --> run
```

**Steps in plain language**

| Step | What you do |
|------|----------------|
| 1. Install | `uv sync` (creates the virtual env and installs dependencies). |
| 2. Configure | Copy `.env.example` to `.env` and set your OpenAI-compatible API key and base URL (see [Configuration](#configuration-env)). |
| 3. Data | Ensure `yelp_dataset/` has `item.json`, `review.json`, `user.json`, and that `yelp_tasks/` / `yelp_groundtruth/` match what you want to evaluate. |
| 4. Smoke test | `uv run python run_simulator_test.py --mock` (no API calls). |
| 5. Real run | `uv run python run_simulator_test.py` (uses your LLM). Tune `SIMULATOR_NUM_TASKS`, threading, and `CREWAI_PROCESS_MODE` via `.env`. |
| 6. Optional | Prebuild a CrewAI knowledge index with `scripts/index_knowledge.py` if you use `CREWAI_ENABLE_KNOWLEDGE=true`. |

---

## Configuration (.env)

1. Copy the template and fill in secrets locally (the real `.env` file is gitignored):

   ```bash
   cp .env.example .env
   ```

   On Windows PowerShell you can use `Copy-Item .env.example .env`.

2. **Required for non-mock runs**: `OPENAI_API_KEY`, and usually `OPENAI_API_BASE` / `OPENAI_BASE_URL` pointing at your provider (NVIDIA NIM, OpenAI, Minimax-compatible host, etc.). Model IDs often appear both here and in `config/agents.yaml`.

3. **Common toggles** (all optional; defaults are in `.env.example`):

   | Variable | Role |
   |----------|------|
   | `CREWAI_PROCESS_MODE` | `sequential` (default), `collaborative`, or `hierarchical` — selects which crew class runs in `AgentSocietyServingFlow`. |
   | `CREWAI_ENABLE_KNOWLEDGE` | `true` to enable Chroma/RAG in `SimulationCrew`; default `false`. |
   | `CREWAI_KNOWLEDGE_FILE` | JSON path for knowledge when RAG is enabled. |
   | `CREWAI_USE_PREBUILT_INDEX` | When RAG is on, skip re-embedding if the index already exists. |
   | `CREWAI_STORAGE_DIR` | Where CrewAI/Chroma stores local state (project-specific folder name). |
   | `CREWAI_EMBEDDER_*` | Embedder for `scripts/index_knowledge.py`. |
   | `SIMULATOR_NUM_TASKS` | Number of tasks to run, or `all` / `full` for every file in the task directory. |
   | `SIMULATOR_THREADING` / `SIMULATOR_MAX_WORKERS` | Parallelism inside `Simulator.run_simulation`. |
   | `GROQ_API_KEY`, `SERPER_API_KEY`, `COHERE_API_KEY` | Only if you add tools or agents that use these providers. |

Do **not** commit `.env` or real API keys. Use `.env.example` as the shared template.

---

## Directory Structure

### 1. **`websocietysimulator/`**  
This is the core library containing all source code required for the competition.

- **`agents/`**: Contains base agent classes (`SimulationAgent`, `RecommendationAgent`) and their abstractions. Participants must extend these classes for their implementations.
- **`task/`**: Defines task structures for each track (`SimulationTask`, `RecommendationTask`).
- **`llm/`**: Contains base LLM client classes (`DeepseekLLM`, `OpenAILLM`).
- **`tools/`**: Includes utility tools:
  - `InteractionTool`: A utility for interacting with the Yelp dataset during simulations.
  - `EvaluationTool`: Provides comprehensive metrics for both recommendation (HR@1/3/5) and simulation tasks (RMSE, sentiment analysis).
- **`simulator.py`**: The main simulation framework, which handles task and groundtruth setting, evaluation and agent execution.

### 2. **`example/`**  
Contains usage examples of the `websocietysimulator` library. Includes sample agents and scripts to demonstrate how to load scenarios, set agents, and evaluate them.

### 3. **`data_process.py`**  
A script to process the raw Yelp dataset into the required format for use with the `websocietysimulator` library. This script ensures the dataset is cleaned and structured correctly for simulations.

---

## Quick Start

### 1. Installation

This CrewAI Sandbox version exclusively utilizes [Astral `uv`](https://github.com/astral-sh/uv) to manage all dependencies. This ensures a blazingly fast and conflict-free "out-of-the-box" experience without relying on complex Anaconda or Poetry configurations.

1. Clone the repository:
   ```bash
   git clone https://github.com/yuchieh/AgentSocietyChallenge_w_CrewAI.git
   cd AgentSocietyChallenge_w_CrewAI
   ```

2. Synchronize all dependencies via `uv`:
   ```bash
   # If you haven't installed uv yet: curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   ```
   *(This step strictly installs both the underlying Simulator's requirements, e.g. pandas/nltk, and the latest CrewAI suite into an isolated virtual environment.)*

3. Verify the Sandbox setup (Mock Mode):
   ```bash
   uv run python run_simulator_test.py --mock
   ```
   *If the environment is set up correctly, this will simulate the CrewAI agents using a mocked LLM (zero token cost) and print a successful JSON evaluation score.* By default it loads **`yelp_dataset`** and runs only the first **5** tasks from **`yelp_tasks`** / **`yelp_groundtruth`**; set environment variable `SIMULATOR_NUM_TASKS=all` to run every task in that folder.

4. Connect to a real LLM:
   Copy `.env.example` to `.env`, set `OPENAI_API_KEY` and your provider base URL (`OPENAI_API_BASE` / `OPENAI_BASE_URL`). See [Configuration (.env)](#configuration-env) for the full list of variables.

   Then run the full simulation without the mock flag:
   ```bash
   uv run python run_simulator_test.py
   ```
   *This mode consumes real tokens as the multi-agent system actively queries the LLM and the Vector Embedding spaces to produce accurate predictions.*

### 1.1 Optional: Prebuild and Reuse CrewAI Knowledge Index

If you want to avoid recomputing embeddings every run, prebuild and persist a CrewAI knowledge index once, then reuse it (set `CREWAI_ENABLE_KNOWLEDGE=true` and the `CREWAI_*` variables in `.env` — see [.env.example](.env.example)):

```bash
uv run python scripts/index_knowledge.py --knowledge-file yelp_dataset/item.json --collection-name crew
uv run python scripts/check_knowledge_index.py --collection-name crew
```

With `CREWAI_USE_PREBUILT_INDEX=true`, `src/crews/simulation_crew.py` can skip re-ingestion when the collection already exists.

---

### 1.2 Optional: Switch Crew Process Mode

Set `CREWAI_PROCESS_MODE` in `.env` to `sequential` (default), `collaborative`, or `hierarchical`. This is read in `src/flows/serving_flow.py` and selects `SimulationCrew`, `CollaborativeSingleTaskCrew`, or `HierarchicalManagerCrew` respectively.

Standalone pattern runners:

```bash
uv run python scripts/run_collaborative_single_task.py --mock
uv run python scripts/run_hierarchical_manager.py --mock
```

---

### 2. Data Preparation

1. Download the raw dataset from the Yelp[1], Amazon[2] or Goodreads[3].
2. Run the `data_process.py` script to process the dataset:
   ```bash
   python data_process.py --input <path_to_raw_dataset> --output <path_to_processed_dataset>
   ```
- Check out the [Data Preparation Guide](./tutorials/data_preparation.md) for more information.
- **NOTICE: You Need at least 16GB RAM to process the dataset.**

---

### 3. Organize Your Data

Ensure the dataset is organized in a directory structure similar to this:

```
<your_dataset_directory>/
├── item.json
├── review.json
├── user.json
```

You can name the dataset directory whatever you prefer (e.g., `dataset/`).

---

### 4. Develop Your Agent

Create a custom agent by extending either `SimulationAgent` or `RecommendationAgent`. Refer to the examples in the `example/` directory. Here's a quick template:

```python
from websocietysimulator.agent import SimulationAgent

class MySimulationAgent(SimulationAgent):
    def workflow(self):
        # The simulator will automatically set the task for your agent. You can access the task by `self.task` to get task information.
        print(self.task)

        # You can also use the `interaction_tool` to get data from the dataset.
        # For example, you can get the user information by `interaction_tool.get_user(user_id="example_user_id")`.
        # You can also get the item information by `interaction_tool.get_item(item_id="example_item_id")`.
        # You can also get the reviews by `interaction_tool.get_reviews(review_id="example_review_id")`.
        user_info = interaction_tool.get_user(user_id="example_user_id")

        # Implement your logic here
        
        # Finally, you need to return the result in the format of `stars` and `review`.
        # For recommendation track, you need to return a candidate list of items, in which the first item is the most recommended item.
        stars = 4.0
        review = "Great experience!"
        return stars, review
```

- Check out the [Tutorial](./tutorials/agent_development.md) for Agent Development.
- Baseline User Behavior Simulation Agent: [Baseline User Behavior Simulation Agent](./example/ModelingAgent_baseline.py).
- Baseline Recommendation Agent: [Baseline Recommendation Agent](./example/RecAgent_baseline.py).
---

### 5. Evaluation your agent with training data

Run the simulation using the provided `Simulator` class:

```python
from websocietysimulator import Simulator
from my_agent import MySimulationAgent

# Initialize Simulator
simulator = Simulator(data_dir="yelp_dataset", device="auto", cache=False)
# The cache parameter controls whether to use cache for interaction tool.
# If you want to use cache, you can set cache=True. When using cache, the simulator will only load data into memory when it is needed, which saves a lot of memory.
# If you want to use normal interaction tool, you can set cache=False. Notice that, normal interaction tool will load all data into memory at the beginning, which needs a lot of memory (20GB+).

# Load scenarios
simulator.set_task_and_groundtruth(task_dir="yelp_tasks", groundtruth_dir="yelp_groundtruth")

# Set your custom agent
simulator.set_agent(MySimulationAgent)

# Set LLM client
simulator.set_llm(DeepseekLLM(api_key="Your API Key"))

# Run evaluation
# If you don't set the number of tasks, the simulator will run all tasks.
agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)

# Evaluate the agent
evaluation_results = simulator.evaluate()
```
- If you want to use your own LLMClient, you can easily implement it by inheriting the `LLMBase` class. Refer to the [Tutorial](./tutorials/agent_development.md) for more information.

---

### 6. Submit your agent
- You should register your team firstly in the competition homepage ([Homepage](https://tsinghua-fib-lab.github.io/AgentSocietyChallenge)).
- Submit your solution through the submission button at the specific track page. (the submission button is at the top right corner of the page)
  - [User Modeling Track](https://tsinghua-fib-lab.github.io/AgentSocietyChallenge/pages/behavior-track.html)
  - [Recommendation Track](https://tsinghua-fib-lab.github.io/AgentSocietyChallenge/pages/recommendation-track.html)
  - Please register your team first.
  - When you submit your agent, please carefully **SELECT the TRACK you want to submit to.**
- **The content of your submission should be a .py file containing your agent (Only one `{your_team}.py` file without evaluation code).**
- Example submissions:
  - For Track 1: [submission_1](example/trackOneSubmission_example.zip)
  - For Track 2: [submission_2](example/trackTwoSubmission_example.zip)

---

## Introduction to the `InteractionTool`

The `InteractionTool` is the core utility for interacting with the dataset. It provides an interface for querying user, item, and review data.

### Functions

- **Get User Information**:
  Retrieve user data by user ID or current scenario context.
  ```python
  user_info = interaction_tool.get_user(user_id="example_user_id")
  ```

- **Get Item Information**:
  Retrieve item data by item ID or current scenario context.
  ```python
  item_info = interaction_tool.get_item(item_id="example_item_id")
  ```

- **Get Reviews**:
  Fetch reviews related to a specific item or user, filtered by time.
  ```python
  reviews = interaction_tool.get_reviews(review_id="example_review_id")  # Fetch a specific review
  reviews = interaction_tool.get_reviews(item_id="example_item_id")  # Fetch all reviews for a specific item
  reviews = interaction_tool.get_reviews(user_id="example_user_id")  # Fetch all reviews for a specific user
  ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

[1] Yelp Dataset: https://www.yelp.com/dataset

[2] Amazon Dataset: https://amazon-reviews-2023.github.io/

[3] Goodreads Dataset: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home