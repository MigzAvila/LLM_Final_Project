"""run_simulator_test.py — AgentSociety + CrewAI integration test runner.

Data source: yelp_dataset + yelp_tasks + yelp_groundtruth (the official Yelp track).
By default runs only the first SIMULATOR_NUM_TASKS tasks (env var, default 5);
set it to "all" to run the full task list.

Two modes:
  1. Real LLM (default): inference via the configured OPENAI-compatible endpoint
  2. Mock mode (--mock): patches litellm.completion with a fake response

Examples:
  uv run python run_simulator_test.py              # real LLM
  uv run python run_simulator_test.py --mock       # mock mode
  $env:SIMULATOR_NUM_TASKS = "all"                 # PowerShell: run every task
  $env:SIMULATOR_THREADING = "true"                # opt back into threading
  $env:SIMULATOR_MAX_WORKERS = "1"                 # workers when threading on
"""
import json
import logging
import os
import sys

from dotenv import load_dotenv
from litellm import ModelResponse

load_dotenv()


def _simulator_num_tasks() -> int | None:
    raw = os.getenv("SIMULATOR_NUM_TASKS", "5").strip().lower()
    if raw in ("", "all", "none", "full"):
        return None
    return int(raw)


def _env_truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _shorten(text: str, limit: int = 220) -> str:
    """Single-line preview of a (possibly multi-line) review."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: max(0, limit - 1)].rstrip() + "…"


def _print_groundtruth_comparison(outputs: list, groundtruth: list) -> None:
    """Print a per-task side-by-side comparison of predictions vs ground truth.

    Predictions whose task timed out (`None` entry) are still shown so the user
    can see exactly which slots failed.
    """
    pair_count = min(len(outputs), len(groundtruth))
    if pair_count == 0:
        print("(no paired outputs to compare)")
        return

    star_deltas: list[float] = []
    print("\n🆚 Prediction vs Ground Truth")
    print("=" * 80)
    for idx in range(pair_count):
        sim = outputs[idx]
        gt = groundtruth[idx] or {}

        if sim and isinstance(sim, dict) and sim.get("output"):
            task = sim.get("task", {}) or {}
            output = sim["output"] or {}
            user_id = task.get("user_id", "?")
            item_id = task.get("item_id", "?")
            pred_stars = output.get("stars")
            pred_review = output.get("review", "")
        else:
            user_id = (sim.get("task", {}) if isinstance(sim, dict) else {}).get("user_id", "?")
            item_id = (sim.get("task", {}) if isinstance(sim, dict) else {}).get("item_id", "?")
            pred_stars = None
            pred_review = "(no prediction — task returned null)"

        gt_stars = gt.get("stars")
        gt_review = gt.get("review") or gt.get("text") or ""

        if isinstance(pred_stars, (int, float)) and isinstance(gt_stars, (int, float)):
            delta = float(pred_stars) - float(gt_stars)
            star_deltas.append(abs(delta))
            delta_str = f"Δ={delta:+.1f}"
        else:
            delta_str = "Δ=n/a"

        pred_stars_str = (
            f"{float(pred_stars):.1f}" if isinstance(pred_stars, (int, float)) else "—"
        )
        gt_stars_str = (
            f"{float(gt_stars):.1f}" if isinstance(gt_stars, (int, float)) else "—"
        )

        print(f"\n[{idx + 1}/{pair_count}] user={user_id} | item={item_id}")
        print(f"  ★ stars  pred={pred_stars_str}  gt={gt_stars_str}  {delta_str}")
        print(f"  📝 pred : {_shorten(pred_review)}")
        print(f"  🎯 gt   : {_shorten(gt_review)}")

    if star_deltas:
        mae = sum(star_deltas) / len(star_deltas)
        exact = sum(1 for d in star_deltas if d == 0)
        within_1 = sum(1 for d in star_deltas if d <= 1.0)
        print("\n" + "-" * 80)
        print(
            f"Summary: MAE={mae:.2f}  "
            f"exact={exact}/{len(star_deltas)}  "
            f"within 1 star={within_1}/{len(star_deltas)}"
        )
    print("=" * 80)


# ----------------------------------------------------------------------
# [Mode] Mock vs. Real LLM
# ----------------------------------------------------------------------
USE_MOCK = "--mock" in sys.argv

if USE_MOCK:
    from unittest.mock import patch

    def fake_completion(*args, **kwargs):
        return ModelResponse(
            id="mock-id",
            model="gpt-4",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"stars": 4.0, "review": "[Mocked LLM] Solid spot, would visit again."}',
                    },
                }
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    # No knowledge indexing is needed in mock mode.
    os.environ.pop("CREWAI_KNOWLEDGE_FILE", None)
    os.environ.pop("CREWAI_KNOWLEDGE_JSON", None)
    os.environ.pop("CREWAI_ENABLE_KNOWLEDGE", None)

    patcher = patch("litellm.completion", side_effect=fake_completion)
    patcher.start()
    print("⚙️  Mode: Mock LLM (no tokens consumed)")
else:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    print("⚙️  Mode: Real LLM (NVIDIA NIM)")
    print(f"🔑 API Key: {'✅ set' if api_key else '❌ missing'}")
    print(f"🌐 Base URL: {api_base or '❌ missing'}")


from websocietysimulator import Simulator
from crewai_simulation_agent import CrewAISimulationAgent

logging.basicConfig(level=logging.INFO)

print("\n" + "=" * 60)
print("🚀 AgentSociety CrewAI integration test (End-to-End)")
print("=" * 60)

try:
    print(">>> Loading Yelp dataset (yelp_dataset)...")
    simulator = Simulator(data_dir="yelp_dataset", device="cpu", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir="yelp_tasks", groundtruth_dir="yelp_groundtruth"
    )
    simulator.set_agent(CrewAISimulationAgent)

    print("\n⚙️  Running inference...")
    n_tasks = _simulator_num_tasks()
    print(f">>> Tasks to run: {'ALL' if n_tasks is None else n_tasks} (SIMULATOR_NUM_TASKS)")

    enable_threading = _env_truthy(os.getenv("SIMULATOR_THREADING", "false"))
    max_workers = int(os.getenv("SIMULATOR_MAX_WORKERS", "1"))
    print(
        f">>> Threading: {'ON' if enable_threading else 'OFF'} "
        f"(max_workers={max_workers})"
    )

    outputs = simulator.run_simulation(
        number_of_tasks=n_tasks,
        enable_threading=enable_threading,
        max_workers=max_workers if enable_threading else None,
    )

    print("\n🏆 Engine output:")
    print("-" * 60)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    print("-" * 60)

    # Snapshot groundtruth BEFORE evaluate() — it truncates the list in place
    # to min(sim_count, gt_count).
    gt_snapshot = list(simulator.groundtruth_data[: len(outputs)])
    _print_groundtruth_comparison(outputs, gt_snapshot)

    print("\n📊 Calling official evaluator (simulator.evaluate())...")
    evaluation_results = simulator.evaluate()
    print("💡 Competition metrics:")
    print(json.dumps(evaluation_results, indent=2, ensure_ascii=False))

    print("\n✅ Integration test complete.")

except Exception as e:
    print(f"\n❌ Test aborted: {e}")
    import traceback

    traceback.print_exc()
