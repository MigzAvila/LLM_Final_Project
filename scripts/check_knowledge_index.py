from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.knowledge.prebuilt_source import (
    collection_document_count,
    default_persist_directory,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check persistent CrewAI knowledge index status."
    )
    parser.add_argument(
        "--collection-name",
        default="crew",
        help="CrewAI collection base name (stored as knowledge_<name>).",
    )
    parser.add_argument(
        "--storage-dir",
        default=os.getenv("CREWAI_STORAGE_DIR", "").strip(),
        help="Optional CREWAI_STORAGE_DIR override.",
    )
    args = parser.parse_args()

    if args.storage_dir:
        os.environ["CREWAI_STORAGE_DIR"] = args.storage_dir

    collection = f"knowledge_{args.collection_name}"
    count = collection_document_count(collection_name=collection)
    print(f"Storage dir: {default_persist_directory()}")
    print(f"Collection: {collection}")
    print(f"Document count: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
