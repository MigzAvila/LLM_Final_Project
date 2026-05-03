from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from crewai.knowledge.knowledge import Knowledge

from src.knowledge.prebuilt_source import (
    PrebuiltTextKnowledgeSource,
    default_persist_directory,
)

logger = logging.getLogger(__name__)


def _sync_openai_compatible_base_url() -> None:
    openai_api_base = os.getenv("OPENAI_API_BASE", "").strip()
    if openai_api_base and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = openai_api_base


def _resolve_embedder(
    provider: str | None,
    model: str | None,
) -> dict[str, object] | None:
    """Build a CrewAI embedder spec.

    Defaults to sentence-transformer to avoid relying on an OpenAI-compatible
    embeddings endpoint (e.g. NVIDIA NIM), which may not host the default
    OpenAI embedding model and yields a 404 on upsert.
    """
    selected_provider = (provider or os.getenv("CREWAI_EMBEDDER_PROVIDER") or "sentence-transformer").strip()
    if not selected_provider:
        return None

    config: dict[str, object] = {}
    selected_model = (model or os.getenv("CREWAI_EMBEDDER_MODEL") or "").strip()
    if selected_model:
        config["model_name"] = selected_model
    elif selected_provider == "sentence-transformer":
        config["model_name"] = "all-MiniLM-L6-v2"
    return {"provider": selected_provider, "config": config}


def _is_embedding_function_conflict(error: Exception) -> bool:
    message = str(error)
    return (
        "embedding function already exists in the collection configuration"
        in message.lower()
        and "embedding function conflict" in message.lower()
    )


def _count_via_knowledge(knowledge: Knowledge, full_collection_name: str) -> int:
    """Count documents using CrewAI's existing storage client.

    Avoids constructing a second `PersistentClient` (which would conflict with
    the Chroma system already initialized by CrewAI for the same path).
    """
    storage = knowledge.storage
    if storage is None:
        return 0
    try:
        client = storage._get_client()
        collection = client.get_or_create_collection(
            collection_name=full_collection_name
        )
        return int(collection.count())
    except Exception:
        return 0


def _reset_existing_collection(knowledge: Knowledge, full_collection_name: str) -> None:
    """Delete the persisted collection using the same client/settings CrewAI built.

    Spawning a second `chromadb.PersistentClient` with default settings clashes
    with CrewAI's already-initialized Chroma system for the same path. Reuse
    the storage's existing client to stay within one Chroma system instance.
    """
    storage = knowledge.storage
    if storage is None:
        return

    try:
        storage.reset()
        return
    except Exception:
        pass

    try:
        client = storage._get_client()
        client.delete_collection(collection_name=full_collection_name)
    except Exception:
        # Collection may not exist or already cleaned up; safe to ignore.
        pass


def build_index(
    knowledge_file: Path,
    collection_name: str,
    force_reindex: bool,
    embedder: dict[str, object] | None = None,
) -> int:
    source = PrebuiltTextKnowledgeSource(
        file_path=knowledge_file,
        collection_name=collection_name,
        skip_if_index_exists=not force_reindex,
    )
    knowledge_kwargs: dict[str, object] = {
        "sources": [source],
        "collection_name": collection_name,
    }
    if embedder is not None:
        knowledge_kwargs["embedder"] = embedder
    knowledge = Knowledge(**knowledge_kwargs)
    full_collection_name = f"knowledge_{collection_name}"
    try:
        knowledge.add_sources()
    except ValueError as error:
        if not _is_embedding_function_conflict(error):
            raise
        logger.warning(
            "Embedding function conflict detected for %s. "
            "Deleting and rebuilding collection.",
            full_collection_name,
        )
        _reset_existing_collection(knowledge, full_collection_name)
        knowledge.add_sources()
    return _count_via_knowledge(knowledge, full_collection_name)


def main() -> int:
    _sync_openai_compatible_base_url()

    parser = argparse.ArgumentParser(
        description="Build or reuse a persistent CrewAI knowledge index."
    )
    parser.add_argument(
        "--knowledge-file",
        default=os.getenv("CREWAI_KNOWLEDGE_FILE", "yelp_dataset/item.json"),
        help="Path to text/JSON file used as crew knowledge source.",
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
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Always run ingestion, even when an index already exists.",
    )
    parser.add_argument(
        "--embedder-provider",
        default=os.getenv("CREWAI_EMBEDDER_PROVIDER", "").strip() or None,
        help=(
            "CrewAI embedder provider (e.g. sentence-transformer, openai, cohere, "
            "huggingface, ollama). Defaults to sentence-transformer."
        ),
    )
    parser.add_argument(
        "--embedder-model",
        default=os.getenv("CREWAI_EMBEDDER_MODEL", "").strip() or None,
        help="Model name for the chosen embedder provider.",
    )
    args = parser.parse_args()

    if args.storage_dir:
        os.environ["CREWAI_STORAGE_DIR"] = args.storage_dir

    knowledge_file = Path(args.knowledge_file).expanduser().resolve()
    if not knowledge_file.exists():
        raise FileNotFoundError(f"Knowledge file not found: {knowledge_file}")

    embedder = _resolve_embedder(args.embedder_provider, args.embedder_model)

    count = build_index(
        knowledge_file=knowledge_file,
        collection_name=args.collection_name,
        force_reindex=args.force_reindex,
        embedder=embedder,
    )

    print(f"Storage dir: {default_persist_directory()}")
    print(f"Collection: knowledge_{args.collection_name}")
    print(f"Document count: {count}")
    if embedder is not None:
        print(f"Embedder: {embedder['provider']} (config={embedder.get('config')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
