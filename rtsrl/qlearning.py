from typing import Dict, List, Tuple

import numpy as np
import utils
from utils import IDLE_TASK_ID, Task

filename = "data/taskset3.txt"
tasks = utils.read_tasks(filename)
utils.assert_under_constraints(tasks)
print(f"TASKS: {tasks}")


class Job:
    def __init__(self, task: Task, t: int):
        self.task = task

        self.deadline = t + task.deadline
        self.exectime_remaining = task.exectime
        self.time_until_deadline = task.deadline

    def do_job(self):
        self.exectime_remaining -= 1
        self.time_until_deadline -= 1

    def idle_job(self):
        self.time_until_deadline -= 1

    def is_done(self) -> bool:
        return self.exectime_remaining <= 0

    def quantize(self, val: int, quanta_base: int) -> int:
        return max(int(quanta_base * (val // quanta_base)), -4)

    def slack(self, quanta_base: int) -> int:
        if self.is_done():
            return -4
        else:
            return self.quantize(
                self.time_until_deadline - self.exectime_remaining, quanta_base
            )


class TaskSchedulingEnvironment:
    def __init__(self, tasks: List[Task], n_quanta: int):
        self.tasks = tasks

        self.max_deadline = max(t.deadline for t in tasks)
        self.quanta_base = self.max_deadline // n_quanta
        self.n_states = (self.quanta_base + 1) ** len(tasks)
        self.n_actions = len(tasks)

        # TODO: ensure task ids are ordered 0 to 4
        self.jobs = {task.id: [Job(task, 0)] for task in tasks}
        self.state = self.reset_state()

        self.state_map: Dict[Tuple[int, ...], int] = dict()
        self.state_i = 0

    def get_state_i(self, state: Tuple[int, ...]) -> int:
        if state not in self.state_map:
            self.state_map[state] = self.state_i
            self.state_i += 1
        return self.state_map[state]

    def reset_state(self) -> Tuple[int, ...]:
        # define state as (time_until_deadline - exectime_remaining) for each task
        state = tuple(Job(task, 0).slack(self.quanta_base) for task in self.tasks)
        return state

    def no_jobs_to_do(self) -> bool:
        return all(jobs[0].is_done() for jobs in self.jobs.values())

    def sample(self) -> int:
        # only choose action for task that has positive exectime_remaining
        choices = [
            task_id
            for task_id, jobs in self.jobs.items()
            if jobs[0].exectime_remaining > 0
        ]
        return np.random.choice(choices)

    def step(self, action, i) -> Tuple[int, int]:

        reward = 0

        # register new job instances on task periods
        if i != 0:
            for task in self.tasks:
                if i % task.period == 0:
                    task_jobs = self.jobs[task.id]
                    if task_jobs[0].is_done():
                        task_jobs.pop(0)
                    task_jobs.append(Job(task, i))

        if action == IDLE_TASK_ID:
            # no tasks to do, return same state with no reward
            return self.get_state_i(self.state), reward

        if self.jobs[action][0].is_done():
            # negative reward for trying to do action that's already done
            reward = -10
            return self.get_state_i(self.state), reward

        # compute action (do first-in job for task and idle for rest)
        for task_id, jobs in self.jobs.items():
            if task_id == action:
                jobs[0].do_job()
                for job in self.jobs[action][1:]:
                    job.idle_job()
            else:
                for job in self.jobs[task_id]:
                    job.idle_job()

        # cleanup and compute rewards
        action_jobs = self.jobs[action]

        if action_jobs[0].is_done():
            reward = 3
            if len(action_jobs) > 1:
                action_jobs.pop(0)

        for jobs in self.jobs.values():
            if not jobs[0].is_done() and jobs[0].slack(self.quanta_base) < 0:
                # -3 for each tasks with negative slack
                reward -= 3

        # max slack for jobs that are done
        self.state = tuple(
            jobs[0].slack(self.quanta_base)
            if not jobs[0].is_done()
            else self.max_deadline
            for task_id, jobs in self.jobs.items()
        )
        return self.get_state_i(self.state), reward


env = TaskSchedulingEnvironment(tasks=tasks, n_quanta=5)


# Initialize the Q-table to 0
Q_table = np.zeros((env.n_states, env.n_actions))

n_episodes = 10000  # number of episode we will run
n_iter_episode = utils.get_lcm_period(tasks)
exploration_proba = 1  # initialize the exploration probability to 1
decay = 0.001  # exploration decreasing decay for exponential decreasing
min_exploration_proba = 0.01  # minimum of exploration proba
gamma = 0.99  # discounted factor
lr = 0.1  # learning rate


rewards_per_episode: List[int] = list()

# iterate over episodes
for t in range(n_episodes):
    # initialize the first state of the episode
    current_state = env.get_state_i(env.reset_state())
    done = False

    # sum the rewards that the agent gets from the environment
    total_episode_reward = 0

    if t % 250 == 0:
        print(f"epi: {t}")

    for i in range(n_iter_episode):
        if env.no_jobs_to_do():
            action = IDLE_TASK_ID
        elif np.random.random() < exploration_proba:
            # exploration of random action
            action = env.sample()
        else:
            # exploitation of currently optimal action
            action = np.argmax(Q_table[current_state, :])

        # Environment runs the chosen action and returns the next state and reward
        next_state, reward = env.step(action, i)

        # Update Q-table using the Q-learning iteration
        if action != IDLE_TASK_ID:
            Q_table[current_state, action] = (1 - lr) * Q_table[
                current_state, action
            ] + lr * (reward + gamma * max(Q_table[next_state, :]))

        total_episode_reward = total_episode_reward + reward
        current_state = next_state

    # Update the exploration proba using exponential decay formula
    exploration_proba = max(min_exploration_proba, np.exp(-decay * t))

    rewards_per_episode.append(total_episode_reward)


print("Mean reward per thousand episodes")
for i in range(10):
    print(
        f"{(i+1)*1000}: mean espiode reward: {np.mean(rewards_per_episode[1000*i:1000*(i+1)])}"
    )
