import os
import sys
from collections import Counter
from statistics import mean
from typing import Any, Iterable

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from websocietysimulator.agent import SimulationAgent
from src.flows.serving_flow import AgentSocietyServingFlow, InferenceState
from src.tools.interaction_tool_wrapper import inject_simulator_tool


# Bound the data we ship into the LLM prompt so we never blow up the context
# window (a single Yelp user can have hundreds of reviews).
MAX_HISTORY_REVIEWS = 12
MAX_REVIEW_TEXT_CHARS = 280
MAX_ITEM_CATEGORIES_CHARS = 240


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Read a key from a dict-or-object without raising."""
    if obj is None:
        return default
    for key in keys:
        if isinstance(obj, dict):
            if key in obj and obj[key] is not None:
                return obj[key]
        else:
            value = getattr(obj, key, None)
            if value is not None:
                return value
    return default


def _summarize_user(user: Any) -> str:
    if not user:
        return "User profile is unavailable. Treat the user as an unknown reviewer."

    name = _safe_get(user, "name", default="Unknown")
    review_count = _safe_get(user, "review_count", default="?")
    yelping_since = _safe_get(user, "yelping_since", default="?")
    avg_stars = _safe_get(user, "average_stars", "avg_stars", default=None)
    avg_str = f"{float(avg_stars):.2f}" if avg_stars is not None else "?"
    useful = _safe_get(user, "useful", default=0)
    funny = _safe_get(user, "funny", default=0)
    cool = _safe_get(user, "cool", default=0)
    fans = _safe_get(user, "fans", default=0)
    elite = _safe_get(user, "elite", default="") or "none"

    return (
        f"name={name}; review_count={review_count}; yelping_since={yelping_since}; "
        f"average_stars={avg_str}; useful={useful}; funny={funny}; cool={cool}; "
        f"fans={fans}; elite={elite}"
    )


def _summarize_item(item: Any) -> str:
    if not item:
        return "Item details are unavailable. Do not invent attributes."

    fields: list[str] = []
    name = _safe_get(item, "name")
    if name:
        fields.append(f"name={name}")
    city = _safe_get(item, "city")
    state = _safe_get(item, "state")
    if city or state:
        fields.append(f"location={city or '?'}, {state or '?'}")
    stars = _safe_get(item, "stars")
    if stars is not None:
        fields.append(f"avg_stars={stars}")
    review_count = _safe_get(item, "review_count")
    if review_count is not None:
        fields.append(f"review_count={review_count}")
    categories = _safe_get(item, "categories")
    if categories:
        fields.append(f"categories={_truncate(str(categories), MAX_ITEM_CATEGORIES_CHARS)}")
    price = None
    attributes = _safe_get(item, "attributes")
    if isinstance(attributes, dict):
        price = attributes.get("RestaurantsPriceRange2")
        noise = attributes.get("NoiseLevel")
        attire = attributes.get("RestaurantsAttire")
        ambience = attributes.get("Ambience")
        if price:
            fields.append(f"price_range={price}")
        if noise:
            fields.append(f"noise={noise}")
        if attire:
            fields.append(f"attire={attire}")
        if ambience:
            fields.append(f"ambience={_truncate(str(ambience), 120)}")

    return "; ".join(fields) if fields else "Item exists but no descriptive fields are available."


def _rating_distribution(reviews: Iterable[dict]) -> str:
    buckets: Counter[int] = Counter()
    total = 0
    for review in reviews:
        stars = _safe_get(review, "stars")
        if stars is None:
            continue
        try:
            star_int = int(round(float(stars)))
        except (TypeError, ValueError):
            continue
        if 1 <= star_int <= 5:
            buckets[star_int] += 1
            total += 1
    if total == 0:
        return "no rated history"
    parts = [f"{star}*={buckets.get(star, 0)}" for star in range(5, 0, -1)]
    return f"n={total}; " + ", ".join(parts)


def _summarize_history(reviews: list[dict]) -> tuple[str, float | None]:
    if not reviews:
        return ("No prior reviews are available for this user.", None)

    sorted_reviews = sorted(
        reviews,
        key=lambda r: str(_safe_get(r, "date", default="")),
        reverse=True,
    )

    star_values: list[float] = []
    for review in sorted_reviews:
        try:
            star_values.append(float(_safe_get(review, "stars", default=0) or 0))
        except (TypeError, ValueError):
            continue
    user_avg = mean(star_values) if star_values else None

    distribution = _rating_distribution(sorted_reviews)

    sample_lines: list[str] = [
        f"TOTAL_HISTORICAL_REVIEWS={len(sorted_reviews)}",
        f"USER_HISTORICAL_AVERAGE_STARS={user_avg:.2f}" if user_avg is not None else "USER_HISTORICAL_AVERAGE_STARS=unknown",
        f"USER_RATING_DISTRIBUTION: {distribution}",
        "RECENT_REVIEWS (most recent first):",
    ]
    for review in sorted_reviews[:MAX_HISTORY_REVIEWS]:
        stars = _safe_get(review, "stars", default="?")
        date = _safe_get(review, "date", default="?")
        text = _truncate(str(_safe_get(review, "text", default="")), MAX_REVIEW_TEXT_CHARS)
        sample_lines.append(f"- [{stars}* on {date}] {text}")
    return ("\n".join(sample_lines), user_avg)


class CrewAISimulationAgent(SimulationAgent):
    """Adapter connecting AgentSociety's simulator framework to the CrewAI flow.

    The retrieval step is performed deterministically against the simulator's
    interaction tool BEFORE the crew runs, so downstream LLM agents always see
    real Yelp data. This eliminates hallucination, removes one LLM round trip,
    and avoids the singleton-tool race when threads run in parallel.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _resolve_ids(self) -> tuple[str, str]:
        if isinstance(self.task, dict):
            return (
                str(self.task.get("user_id", "") or ""),
                str(self.task.get("item_id", "") or ""),
            )
        return (
            str(getattr(self.task, "user_id", "") or ""),
            str(getattr(self.task, "item_id", "") or ""),
        )

    def _safe_call(self, func, *args, **kwargs):
        if func is None:
            return None
        try:
            return func(*args, **kwargs)
        except Exception:
            return None

    def workflow(self) -> dict:
        user_id, item_id = self._resolve_ids()

        # Make the tool available to optional collaborative/hierarchical paths
        # that still rely on the wrapper. The sequential path no longer depends
        # on it because we pre-fetch below.
        inject_simulator_tool(getattr(self, "interaction_tool", None))

        tool = getattr(self, "interaction_tool", None)
        user = self._safe_call(getattr(tool, "get_user", None), user_id=user_id)
        item = self._safe_call(getattr(tool, "get_item", None), item_id=item_id)
        reviews = self._safe_call(getattr(tool, "get_reviews", None), user_id=user_id) or []

        user_summary = _summarize_user(user)
        item_summary = _summarize_item(item)
        history_summary, user_avg = _summarize_history(reviews)

        fallback_rating = (
            float(user_avg)
            if user_avg is not None
            else float(_safe_get(user, "average_stars", "avg_stars", default=4.0) or 4.0)
        )
        fallback_rating = max(1.0, min(5.0, fallback_rating))

        initial_state = InferenceState(
            user_id=user_id,
            item_id=item_id,
            user_summary=user_summary,
            item_summary=item_summary,
            history_summary=history_summary,
            fallback_rating=fallback_rating,
        )

        flow = AgentSocietyServingFlow(initial_state=initial_state)
        final_state_dict = flow.kickoff()

        return {
            "stars": float(final_state_dict.get("predicted_rating", fallback_rating)),
            "review": str(final_state_dict.get("generated_review", "")) or "No review generated.",
        }
