import heapq
import random
from collections import deque
from typing import Callable, Dict, List, Tuple

from utils import IDLE_TASK_ID, Task

random.seed(1)


def rate_monotonic_schedule(tasks: List[Task], runtime: int) -> List[List[int]]:
    """
    Schedules tasks according to the rate-monotonic scheduling algorithm where the period is inversely proportional to the priority of a task. (i.e. the task with the shortest period has highest priority.)
    """
    req_utilization = sum(task.exectime / task.period for task in tasks)
    n = len(tasks)
    utilization_rm = n * (2 ** (1 / n) - 1)
    assert (
        req_utilization <= utilization_rm
    ), "Tasks are not schedulable using the RM scheduling algorithm."

    prioritized_tasks = [
        task.id for task in sorted(tasks, key=lambda task: task.period)
    ]

    def rate_monotonic_scheduling(todo_tasks: List[Task]) -> int:
        for task_id in prioritized_tasks:
            if task_id in todo_tasks:
                return task_id
        return IDLE_TASK_ID

    return schedule(
        tasks=tasks, runtime=runtime, scheduling_algorithm=rate_monotonic_scheduling
    )


def randomly_schedule(tasks: List[Task], runtime: int) -> List[List[int]]:
    """
    Randomly schedules tasks. Assumes that if there's any task to be worked on, it will work on it rather than stay idle.
    """

    def random_scheduling(todo_tasks: List[Task]) -> int:
        return random.choice(todo_tasks)

    return schedule(
        tasks=tasks, runtime=runtime, scheduling_algorithm=random_scheduling
    )


def schedule(
    tasks: List[Task], runtime: int, scheduling_algorithm: Callable[[List[Task]], int]
) -> List[List[int]]:
    """
    Schedules tasks using the scheduling_algorithm passed in to choose the next task to work on.
    """
    # a min-heap to track when tasks need to be requeued / released
    tasks_queue: List[Tuple[int, Task]] = []  # [(next_time_to_queue, task)]
    for task in tasks:
        heapq.heappush(tasks_queue, (0, task))

    # a list and dict to hold tasks+metadata that can be done at the current time
    todo_tasks = []  # [task_id]
    todo_dict: Dict[int, deque] = {}  # {task_id: deque([deadline, exectime_remaining])}

    # resulting chronological schedule of task_id and duration pairs
    schedule: List[List[int]] = []  # [[task_id, duration]]

    for t in range(runtime):
        # check if need to requeue / release any tasks
        while t == tasks_queue[0][0]:
            task = heapq.heappop(tasks_queue)[1]

            if task.id not in todo_dict:
                todo_dict[task.id] = deque()
                todo_tasks.append(task.id)

            todo_dict[task.id].append([t + task.deadline, task.exectime])

            heapq.heappush(tasks_queue, (t + task.period, task))

        # choose next task to execute from todo_tasks
        if not todo_tasks:
            task_id = IDLE_TASK_ID
        else:
            task_id = scheduling_algorithm(todo_tasks)

        # append to schedule
        if schedule and schedule[-1][0] == task_id:
            schedule[-1][1] += 1
        else:
            schedule.append([task_id, 1])

        # remove 1 unit of exectime from task in todo_dict
        if task_id != IDLE_TASK_ID:
            todo_dict[task_id][0][1] -= 1
            if todo_dict[task_id][0][1] == 0:
                todo_dict[task_id].popleft()
                if len(todo_dict[task_id]) == 0:
                    del todo_dict[task_id]
                    todo_tasks.remove(task_id)

    return schedule
