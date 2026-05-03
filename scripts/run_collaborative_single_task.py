import argparse
import os
import sys
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv
from litellm import ModelResponse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.crews.collaborative_single_task_crew import CollaborativeSingleTaskCrew


def _start_mock_if_needed(use_mock: bool):
    if not use_mock:
        return None

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
                        "content": '{"stars": 4.2, "review": "[Mocked] Balanced and realistic review output."}',
                    },
                }
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    patcher = patch("litellm.completion", side_effect=fake_completion)
    patcher.start()
    return patcher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default="u_demo")
    parser.add_argument("--item-id", default="i_demo")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    patcher = _start_mock_if_needed(args.mock)
    try:
        result = CollaborativeSingleTaskCrew().crew().kickoff(
            inputs={"user_id": args.user_id, "item_id": args.item_id}
        )
        print(result.raw)
        return 0
    finally:
        if patcher:
            patcher.stop()


if __name__ == "__main__":
    raise SystemExit(main())
