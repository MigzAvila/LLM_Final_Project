import os

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from src.knowledge.prebuilt_source import PrebuiltTextKnowledgeSource


def _env_truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sync_openai_compatible_base_url() -> None:
    """Mirror OPENAI_API_BASE -> OPENAI_BASE_URL so CrewAI's embedding stack
    does not silently fall back to api.openai.com."""
    openai_api_base = os.getenv("OPENAI_API_BASE", "").strip()
    if openai_api_base and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = openai_api_base


def _build_optional_knowledge_sources() -> list[PrebuiltTextKnowledgeSource]:
    # Knowledge is opt-in. Default OFF because the sequential pipeline already
    # injects ground-truth Yelp data into the prompts, and a stale persistent
    # collection (sentence_transformer vs openai) caused noisy embedding errors.
    if not _env_truthy(os.getenv("CREWAI_ENABLE_KNOWLEDGE")):
        return []

    _sync_openai_compatible_base_url()

    knowledge_file = os.getenv("CREWAI_KNOWLEDGE_FILE", "").strip()
    if not knowledge_file:
        knowledge_file = os.getenv("CREWAI_KNOWLEDGE_JSON", "").strip()
    if not knowledge_file:
        return []

    skip_if_exists = _env_truthy(os.getenv("CREWAI_USE_PREBUILT_INDEX"))
    return [
        PrebuiltTextKnowledgeSource(
            file_path=knowledge_file,
            collection_name="crew",
            skip_if_index_exists=skip_if_exists,
        )
    ]


@CrewBase
class SimulationCrew():
    """Sequential simulation crew.

    Data retrieval happens deterministically in CrewAISimulationAgent before
    the crew runs. The crew now consists of two LLM agents only: the analyst
    that interprets the prefetched data, and the simulator that emits the
    final {stars, review} JSON. This drops one full LLM round-trip and
    guarantees that every downstream prompt is grounded in real Yelp data.
    """

    agents_config = '../../config/agents.yaml'
    tasks_config = '../../config/tasks.yaml'

    @agent
    def psychological_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['psychological_analyst'],
            verbose=False,
            allow_delegation=False,
            max_iter=2,
        )

    @agent
    def behavior_simulator(self) -> Agent:
        return Agent(
            config=self.agents_config['behavior_simulator'],
            verbose=False,
            allow_delegation=False,
            max_iter=2,
        )

    @task
    def analyze_preference_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_preference_task']
        )

    @task
    def simulate_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['simulate_review_task']
        )

    @crew
    def crew(self) -> Crew:
        knowledge_sources = _build_optional_knowledge_sources()

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            **({"knowledge_sources": knowledge_sources} if knowledge_sources else {}),
        )
