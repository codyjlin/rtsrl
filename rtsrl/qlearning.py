from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import utils
from utils import IDLE_TASK_ID, Task

np.random.seed(0)
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

    def __repr__(self):
        return f"task_id: {self.task.id}, deadline: {self.deadline}, exectime_remaining: {self.exectime_remaining}"


class TaskSchedulingEnvironment:
    def __init__(self, tasks: List[Task], n_quanta: int):
        utils.assert_under_constraints(tasks, max_utilization=1, max_exectime=20)
        self.tasks = tasks

        self.max_deadline = max(t.deadline for t in tasks)
        self.quanta_base = self.max_deadline // n_quanta
        self.n_states = (self.quanta_base + 1) ** len(tasks)
        self.n_actions = len(tasks)

        self.n_iter_episode = utils.get_lcm_period(tasks)
        self.max_reward = sum(self.n_iter_episode / task.period for task in tasks)

        self.reset_state()

    def reset_state(self) -> Tuple[int, ...]:
        # define state as (time_until_deadline - exectime_remaining) for each task
        self.jobs = {task.id: [Job(task, 0)] for task in self.tasks}
        self.state = tuple(Job(task, 0).slack(self.quanta_base) for task in self.tasks)
        return self.state

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

    def step(self, action, i) -> Tuple[Tuple[int, ...], int, int]:

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
            return self.state, reward, -1

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
            reward = 1
            if len(action_jobs) > 1:
                action_jobs.pop(0)

        for jobs in self.jobs.values():
            if (
                not jobs[0].is_done()
                and jobs[0].time_until_deadline - jobs[0].exectime_remaining < 0
            ):
                # -3 for each tasks with negative slack
                reward -= 1
                # print("NEGATIVE_REWARD!!!")

        # find action index that corresponds to action in sorted state before updating to new state
        for j, slack in enumerate(sorted(self.state, reverse=True)):
            if slack == self.state[action]:
                action_i_in_sorted_state = j
                break

        # max slack for jobs that are done
        self.state = tuple(
            jobs[0].slack(self.quanta_base)
            if not jobs[0].is_done()
            else self.max_deadline
            for task_id, jobs in self.jobs.items()
        )

        return tuple(sorted(self.state, reverse=True)), reward, action_i_in_sorted_state


def q_learning(
    n_episodes=10000,
    exploration_proba=1,
    decay=0.0001,
    min_exploration_proba=0.01,
    gamma=0.99,
    lr=0.1,
):
    """
    Args:
        n_iter_episode - number of iterations per episode
        n_episodes - number of episode we will run
        exploration_proba - the initial exploration probability
        decay - exploration decreasing decay for exponential decreasing
        min_exploration_proba - minimum of exploration proba
        gamma - discounted factor
        lr - learning rate
    """
    Q_table: Dict[Tuple[int, ...]] = dict()
    state_count: DefaultDict[Tuple[int, ...], int] = defaultdict(int)
    rewards_per_episode: List[int] = list()

    for t in range(n_episodes):
        # generate new taskset for each episode
        tasks = utils.generate_random_taskset()
        env = TaskSchedulingEnvironment(tasks=tasks, n_quanta=5)
        current_state = tuple(sorted(env.state, reverse=True))  # needs to be sorted
        total_episode_reward = 0

        if t % 250 == 0:
            print(f"epi: {t}, exploration_p: {exploration_proba}")

        for i in range(env.n_iter_episode):
            # count number of times a state is visited
            state_count[current_state] += 1

            if current_state not in Q_table:
                Q_table[current_state] = np.zeros(len(current_state))

            if env.no_jobs_to_do():
                action = IDLE_TASK_ID
            elif np.random.random() < exploration_proba:
                # exploration of random action
                action = env.sample()
            else:
                # exploitation of currently optimal action
                for index in np.argsort(-Q_table[current_state]):

                    # unsort the sorted state in Q_table to find action that index corresponds to
                    for task_id, slack in enumerate(env.state):
                        if (
                            slack == current_state[index]
                            and not env.jobs[task_id][0].is_done()
                        ):
                            action = task_id
                            break

                    if not env.jobs[action][0].is_done():
                        break

            # environment runs the chosen action and returns the next state and reward
            next_state, reward, action_i_in_sorted_state = env.step(action, i)

            # update Q-table using the Q-learning iteration
            if action != IDLE_TASK_ID:

                if next_state not in Q_table:
                    Q_table[next_state] = np.zeros(len(next_state))

                Q_table[current_state][action_i_in_sorted_state] = (1 - lr) * Q_table[
                    current_state
                ][action_i_in_sorted_state] + lr * (
                    reward + gamma * max(Q_table[next_state])
                )

            total_episode_reward = total_episode_reward + reward
            current_state = next_state

        # Update the exploration proba using exponential decay formula
        exploration_proba = max(min_exploration_proba, np.exp(-decay * t))

        rewards_per_episode.append(total_episode_reward / env.max_reward)

    return Q_table, rewards_per_episode, state_count


Q_table, rewards_per_episode, state_count = q_learning()
_, rewards_per_ep_random_sched, _ = q_learning(n_episodes=5000, min_exploration_proba=1)

# Printing number of times each state was visited
for state in sorted(state_count, key=lambda x: sum(x), reverse=True):
    print(state, state_count[state])

# Printing mean rewards per 100 episodes
print("Mean reward per 100 episodes")
for i in range(100):
    print(
        f"{(i+1)*100}: mean episode reward: {np.mean(rewards_per_episode[100*i:100*(i+1)])}"
    )
print(f"Mean reward for random scheduling: {np.mean(rewards_per_ep_random_sched)}")

# Plotting mean rewards per 100 episodes
mean_rewards_per_100 = [
    np.mean(rewards_per_episode[100 * i : 100 * (i + 1)]) for i in range(100)
]

fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
ax.plot(
    np.arange(100),
    np.fromiter(mean_rewards_per_100, dtype=float),
    label="Rewards per episode",
)
ax.set_xlabel("i")
ax.set_ylabel("reward")
plt.show()
