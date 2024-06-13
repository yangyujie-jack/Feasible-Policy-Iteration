import copy

from safety_gymnasium.utils.registration import register


VERSION = 'v0'
ROBOT_NAMES = ('Point', 'Car')
PREFIX = 'Custom'


def __combine(tasks, agents, max_episode_steps):
    """Combine tasks and agents together to register environment tasks."""
    for task_name, task_config in tasks.items():
        for robot_name in agents:
            env_id = f'{PREFIX}{robot_name}{task_name}-{VERSION}'
            combined_config = copy.deepcopy(task_config)
            combined_config.update({'agent_name': robot_name})

            register(
                id=env_id,
                entry_point='fpi.env.builder:CustomBuilder',
                kwargs={'config': combined_config, 'task_id': env_id},
                max_episode_steps=max_episode_steps,
            )


tasks = {'Goal1': {}}
__combine(tasks, ROBOT_NAMES, max_episode_steps=1000)
