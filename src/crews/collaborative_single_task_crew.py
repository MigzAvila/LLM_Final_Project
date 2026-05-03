from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class CollaborativeSingleTaskCrew:
    """Process.Sequential Pattern 2: collaborative single task."""

    agents_config = "../../config/agents.yaml"
    tasks_config = "../../config/tasks_collaborative.yaml"

    @agent
    def data_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config["data_retriever"],
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

    @agent
    def psychological_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["psychological_analyst"],
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

    @agent
    def behavior_simulator(self) -> Agent:
        return Agent(
            config=self.agents_config["behavior_simulator"],
            verbose=True,
            allow_delegation=True,
            max_iter=4,
        )

    @task
    def collaborative_single_task(self) -> Task:
        return Task(config=self.tasks_config["collaborative_single_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
