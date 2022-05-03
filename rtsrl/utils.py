from collections import defaultdict
from math import gcd
from textwrap import dedent
from typing import List, NamedTuple

import numpy as np

np.random.seed(0)

IDLE_TASK_ID = -1


class Task(NamedTuple):
    id: int
    period: int
    exectime: int
    deadline: int


class CLIColors:
    PURPLE = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ENDC = "\033[0m"


def read_tasks(filename: str) -> List[Task]:
    """
    Opens the file at filename and returns a list of tasks. Casts floats to integers by floor.
    """
    with open(filename) as f:
        return [
            Task._make([int(x) for x in line.split(", ")])
            for line in f.readlines()
            if len(line) > 1 and line[0] != "#"
        ]


def generate_random_taskset() -> List[Task]:
    """
    Generates a random taskset and returns the list of tasks.
    Subject to following constraints:
        - len(tasks) <= 5
        - 0 < task.period <= 20
        - 0 < task.exectime <= 5 (TODO: see if this constraint is needed)
        - task.period % 5 == 0
        - task.deadline == task.period
        - utilization == sum(task.exectime / task.period for task in tasks) and <= 1
    """
    len_tasks = np.random.randint(2, 6)
    utilization = 1.1

    while utilization > 1:
        tasks = []
        utilization_target = np.random.uniform(0.8, 1)
        utilization_per_task = (
            np.random.dirichlet(np.ones(len_tasks)) * utilization_target
        )
        for i in range(len_tasks):
            utilization_contribution = utilization_per_task[i]
            period = np.random.choice([5, 10, 15, 20])
            exectime = max(np.floor(utilization_contribution * period), 1)
            tasks.append(
                Task(id=i, period=period, exectime=int(exectime), deadline=period)
            )
        utilization = sum(task.exectime / task.period for task in tasks)
    return tasks


def get_lcm_period(tasks: List[Task]) -> int:
    """
    Returns the least common multiple of the periods from the input list of tasks.
    """
    lcm = 1
    for task in tasks:
        lcm = lcm * task.period // gcd(lcm, task.period)

    return lcm


def assert_is_schedulable(tasks: List[Task]):
    utilization = sum(task.exectime / task.period for task in tasks)
    assert utilization <= 1, "Tasks are not schedulable using any scheduling algorithm."


def assert_under_constraints(
    tasks: List[Task],
    max_tasks=5,
    max_period=20,
    max_exectime=5,
    period_multiple=5,
    max_utilization=0.8,
):
    assert len(tasks) <= max_tasks
    for task in tasks:
        assert 0 < task.period <= max_period
        assert 0 < task.exectime <= max_exectime
        assert task.period % period_multiple == 0
        assert task.deadline == task.period

    utilization = sum(task.exectime / task.period for task in tasks)
    assert (
        utilization <= max_utilization
    ), f"Total utilization is above {max_utilization*100}%"


def compress_schedule(schedule: List[int]) -> List[List[int]]:
    """
    Takes in schedule as long list containing task_ids, representing task worked on for each unit of time. Returns compressed format of schedule as a list of (task_id, duration) pairs.
    """
    compressed_schedule: List[List[int]] = []

    for task_id in schedule:
        if compressed_schedule and compressed_schedule[-1][0] == task_id:
            compressed_schedule[-1][1] += 1
        else:
            compressed_schedule.append([task_id, 1])

    return compressed_schedule


def print_compressed_schedule(schedule: List[List[int]], print_descrip=False):
    """
    Simply prints the schedule line by line as a list of (task_id, duration) pairs, with a task_id of -1 representing idle state.
    """
    print(
        "Printing schedule as a chronological list of (task_id, duration) pairs, with a task_id of -1 representing idle state.\n"
    )

    for job in schedule:
        print(job)


def print_tasks(tasks: List[Task]):
    """
    Prints tasks metadata.
    """
    for task in tasks:
        print(
            dedent(
                f"""\
        Task {task.id}:
            - period: {task.period}
            - exectime: {task.exectime}
            - deadline: {task.deadline}"""
            )
        )


def print_by_task(tasks: List[Task], schedule: List[int], print_descrip=False):
    """
    Prints a line per task to visualize when each task is being worked on. Alternates between yellow and blue to represent the particular task's period boundaries. Red represents any late tasks. (Upper and lower case shown here in example.)
    Example
        for a task with a period of 3 units:
        iiiIIIiiiIII represents not working at all
        wwwWWWwwwWWW represents working all the time
        wwiWIIiwiWWW represents work 2, idle 1, work 1, idle 3, work 1, idle 1, work 3
    """
    task_executions: defaultdict = defaultdict(list)
    task_periods = {task.id: task.period for task in tasks}
    task_exectimes = {task.id: task.exectime for task in tasks}

    for i, task_id in enumerate(schedule):
        if task_id != IDLE_TASK_ID:
            len_idle = i - len(task_executions[task_id])
            task_executions[task_id].extend(["_"] * len_idle)
            task_executions[task_id].append("w")

    if print_descrip:
        print(
            dedent(
                """\
        Printing schedule by task, with the following semantics:
            - i or I represents idle state
            - w or W represents working state
            - changes in lower/upper case represents changes in period
        """
            )
        )

    for task_id in sorted(task_executions.keys()):
        len_idle = len(schedule) - len(task_executions[task_id])
        task_executions[task_id].extend(["_"] * len_idle)

        line = task_executions[task_id]
        period = task_periods[task_id]

        line_list = []
        exectime_remaining = task_exectimes[task_id]
        overdue = False

        for i, x in enumerate(line):
            if (i // period) % 2:
                color = CLIColors.CYAN
            else:
                color = CLIColors.YELLOW

            if i > 0 and i % period == 0:
                if exectime_remaining == 0:
                    exectime_remaining = task_exectimes[task_id]
                    overdue = False
                else:
                    overdue = True

            if x == "w":
                exectime_remaining -= 1
                if overdue and exectime_remaining == 0:
                    color = CLIColors.RED
                    exectime_remaining = task_exectimes[task_id]
                    overdue = False

            if overdue:
                color = CLIColors.RED

            line_list.append(f"{color}{x}{CLIColors.ENDC}")

        print(
            f"Task {task_id} ({period}, {task_exectimes[task_id]}): {''.join(line_list)}"
        )
