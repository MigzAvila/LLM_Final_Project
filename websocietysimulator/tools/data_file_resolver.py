import os
from typing import Dict, List


_DATASET_FALLBACKS: Dict[str, List[str]] = {
    "user.json": ["user_subset.json"],
    "item.json": ["item_subset.json"],
    "review.json": ["review_subset.json", "test_review_subset.json"],
}


def resolve_dataset_file(data_dir: str, filename: str) -> str:
    """
    Return the first readable dataset file path for a logical filename.

    This keeps default behavior (`*.json`) while allowing environments where
    symlinked files are unavailable (common on Windows checkouts) to fall back
    to subset files bundled in the repository.
    """
    candidates = [filename, *_DATASET_FALLBACKS.get(filename, [])]
    search_dirs = [data_dir]

    parent_data_dir = os.path.join(os.path.dirname(os.path.abspath(data_dir)), "data")
    if os.path.abspath(parent_data_dir) != os.path.abspath(data_dir):
        search_dirs.append(parent_data_dir)

    checked_paths = []
    for search_dir in search_dirs:
        for candidate in candidates:
            candidate_path = os.path.join(search_dir, candidate)
            checked_paths.append(candidate_path)
            if os.path.isfile(candidate_path) and os.access(candidate_path, os.R_OK):
                return candidate_path

    checked = ", ".join(checked_paths)
    raise FileNotFoundError(
        f"No readable dataset file found for '{filename}'. Checked: {checked}"
    )
