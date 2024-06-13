import fpi.env.task as tasks
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.builder import Builder
from safety_gymnasium.utils.task_utils import get_task_class_name


class CustomBuilder(Builder):
    def _get_task(self) -> BaseTask:
        """Instantiate a task object."""
        class_name = 'Custom' + get_task_class_name(self.task_id)
        assert hasattr(tasks, class_name), f'Task={class_name} not implemented.'
        task_class = getattr(tasks, class_name)
        task = task_class(config=self.config)

        task.build_observation_space()
        return task
