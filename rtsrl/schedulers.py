import heapq
import random
from collections import deque
from typing import Dict, List, Tuple

from utils import IDLE_TASK_ID, Task

random.seed(1)


def randomly_schedule(tasks: List[Task], runtime: int) -> List[List[int]]:
    """
    Randomly schedules tasks. Assumes that if there's any task to be worked on, it will work on it rather than stay idle.
    """
    # a min-heap to track when tasks need to be requeued / marked ready
    tasks_queue: List[Tuple[int, Task]] = []  # [(next_time_to_queue, task)]
    for task in tasks:
        heapq.heappush(tasks_queue, (0, task))

    # a list and dict to hold tasks+metadata that can be done at the current time
    todo_tasks = []  # [task_id]
    todo_dict: Dict[int, deque] = {}  # {task_id: deque([deadline, exectime_remaining])}

    # resulting chronological schedule of task_id and duration pairs
    schedule: List[List[int]] = []  # [[task_id, duration]]

    for t in range(runtime):
        # check if need to mark any tasks ready
        while t == tasks_queue[0][0]:
            task = heapq.heappop(tasks_queue)[1]

            if task.id not in todo_dict:
                todo_dict[task.id] = deque()
                todo_tasks.append(task.id)

            todo_dict[task.id].append([t + task.deadline, task.exectime])

            heapq.heappush(tasks_queue, (t + task.period, task))

        # randomly choose task to execute from todo_tasks
        if not todo_tasks:
            task_id = IDLE_TASK_ID
        else:
            task_id = random.choice(todo_tasks)

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
