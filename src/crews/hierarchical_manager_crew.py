from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from src.tools.interaction_tool_wrapper import get_interaction_tool


@CrewBase
class HierarchicalManagerCrew:
    """Process.hierarchical with explicit manager agent."""

    agents_config = "../../config/agents.yaml"
    tasks_config = "../../config/tasks_hierarchical.yaml"

    def prediction_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_manager"],
            verbose=True,
            allow_delegation=True,
        )

    @agent
    def data_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config["data_retriever"],
            verbose=True,
            allow_delegation=False,
            tools=[get_interaction_tool()],
        )

    @agent
    def psychological_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["psychological_analyst"],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def behavior_simulator(self) -> Agent:
        return Agent(
            config=self.agents_config["behavior_simulator"],
            verbose=True,
            allow_delegation=False,
        )

    @task
    def hierarchical_predict_task(self) -> Task:
        return Task(config=self.tasks_config["hierarchical_predict_task"])

    @crew
    def crew(self) -> Crew:
        worker_agents = [
            self.data_retriever(),
            self.psychological_analyst(),
            self.behavior_simulator(),
        ]
        return Crew(
            agents=worker_agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self.prediction_manager(),
            verbose=True,
        )
