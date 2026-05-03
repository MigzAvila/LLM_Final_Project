"""CrewAI tool wrapper around the simulator's interaction tool.

Used by the optional collaborative / hierarchical crew variants. The default
sequential crew no longer relies on this wrapper because data is prefetched
deterministically in `CrewAISimulationAgent.workflow()`.
"""

import json
import threading
from typing import Any

from crewai.tools import tool

# Stored in module-level state. Workers in different threads need to access
# the same simulator-provided InteractionTool, so a lock is sufficient: there
# is exactly one InteractionTool per Simulator instance and it is read-only
# from the workers' perspective.
_GLOBAL_INTERACTION_TOOL: Any = None
_LOCK = threading.Lock()


def inject_simulator_tool(tool_instance: Any) -> None:
    global _GLOBAL_INTERACTION_TOOL
    with _LOCK:
        _GLOBAL_INTERACTION_TOOL = tool_instance


def _serialize(payload: Any) -> str:
    if payload is None:
        return "NOT_FOUND"
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(payload)


@tool("Interaction Tool Wrapper")
def interaction_tool_wrapper(query_type: str, target_id: str) -> str:
    """Query AgentSociety's local retrieval tool for historical data.

    query_type must be one of "user", "item", "review_by_user", "review_by_item".
    target_id is the corresponding user_id or item_id.
    Returns a JSON string, or "NOT_FOUND" if no record exists.
    """
    with _LOCK:
        tool_ref = _GLOBAL_INTERACTION_TOOL

    if tool_ref is None:
        return "Error: InteractionTool has not been injected by the Simulator."

    try:
        if query_type == "user":
            return _serialize(tool_ref.get_user(user_id=target_id))
        if query_type == "item":
            return _serialize(tool_ref.get_item(item_id=target_id))
        if query_type == "review_by_user":
            return _serialize(tool_ref.get_reviews(user_id=target_id))
        if query_type == "review_by_item":
            return _serialize(tool_ref.get_reviews(item_id=target_id))
        return (
            "Error: Unknown query_type. Use 'user', 'item', 'review_by_user' "
            "or 'review_by_item'."
        )
    except Exception as exc:
        return f"Error during interaction_tool query: {exc}"


def get_interaction_tool():
    """Return the CrewAI tool instance for use by an Agent."""
    return interaction_tool_wrapper
