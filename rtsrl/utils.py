from collections import defaultdict
from math import gcd
from textwrap import dedent
from typing import List, NamedTuple

IDLE_TASK_ID = -1


class Task(NamedTuple):
    id: int
    period: int
    exectime: int
    deadline: int


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


def print_schedule(schedule: List[List[int]]):
    """
    Simply prints the schedule line by line as a list of (task_id, duration) pairs, with a task_id of -1 representing idle state.
    """
    print(
        "Printing schedule as a chronological list of (task_id, duration) pairs, with a task_id of -1 representing idle state.\n"
    )

    for job in schedule:
        print(job)


def print_by_task(tasks: List[Task], schedule: List[List[int]]):
    """
    Prints a line per task to visualize when each task is being worked on. Alternates between upper and lower case to represent the particular task's period boundaries.
    Example
        for a task with a period of 3 units:
        iiiIIIiiiIII represents not working at all
        wwwWWWwwwWWW represents working all the time
        wwiWIIiwiWWW represents work 2, idle 1, work 1, idle 3, work 1, idle 1, work 3
    """
    task_executions: defaultdict = defaultdict(list)
    task_periods = {task.id: task.period for task in tasks}
    i = 0
    for task_id, duration in schedule:
        if task_id != IDLE_TASK_ID:
            len_idle = i - len(task_executions[task_id])
            task_executions[task_id].extend(["i"] * len_idle)
            task_executions[task_id].extend(["w"] * duration)
        i += duration

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
        len_idle = i - len(task_executions[task_id])
        task_executions[task_id].extend(["i"] * len_idle)

        line = task_executions[task_id]
        period = task_periods[task_id]

        print(
            f"Task {task_id}: {''.join(x.upper() if (i//period)%2 else x for i, x in enumerate(line))}"
        )


def print_stats(schedule: List[List[int]]):
    """
    Prints stats of a schedule:
        - total_working: the amount of time spent working
        - total_idle: the amount of time spent in the idle state
    """
    total_working = sum(y for x, y in schedule if x != -1)
    total_idle = sum(y for x, y in schedule if x == -1)

    print("Printing schedule stats:")

    print(f"total_working: {total_working}")
    print(f"total_idle: {total_idle}")
