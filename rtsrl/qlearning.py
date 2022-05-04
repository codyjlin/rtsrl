from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import utils
from utils import IDLE_TASK_ID, Task

np.random.seed(0)


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
                if (i + 1) % task.period == 0:
                    task_jobs = self.jobs[task.id]
                    if task_jobs[0].is_done():
                        task_jobs.pop(0)
                    task_jobs.append(Job(task, i + 1))

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
                # TODO: definitely a bug here that's causing non-overdue jobs to be punished
                reward -= 1

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
    n_tasksets=4000,
    n_repeat=5,
    exploration_proba=1,
    decay=0.0001,
    min_exploration_proba=0.01,
    gamma=0.99,
    lr=0.1,
    print_status=False,
):
    """
    Trains a QL scheduler and returns the Q_table, along with metadata like the training set, rewards_per_episode, state_count, and rewards_by_utilization during the training process.
    Args:
        n_tasksets - number of tasksets to train on
        n_repeat - number of times each taskset is trained on
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
    rewards_by_utilization: DefaultDict[float, List[float]] = defaultdict(list)

    training_set = [utils.generate_random_taskset() for _ in range(n_tasksets)]
    num_tasksets = len(training_set)

    for repeat_i in range(n_repeat):
        for taskset_i, tasks in enumerate(training_set):
            t = taskset_i + num_tasksets * repeat_i

            env = TaskSchedulingEnvironment(tasks=tasks, n_quanta=5)
            current_state = tuple(sorted(env.state, reverse=True))
            total_episode_reward = 0

            if t % 250 == 0 and print_status:
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

                    Q_table[current_state][action_i_in_sorted_state] = (
                        1 - lr
                    ) * Q_table[current_state][action_i_in_sorted_state] + lr * (
                        reward + gamma * max(Q_table[next_state])
                    )

                total_episode_reward = total_episode_reward + reward
                current_state = next_state

            # Update the exploration proba using exponential decay formula
            exploration_proba = max(min_exploration_proba, np.exp(-decay * t))

            rewards_per_episode.append(total_episode_reward / env.max_reward)

            utilization = sum(task.exectime / task.period for task in tasks)
            rewards_by_utilization[round(utilization * 20) / 20].append(
                total_episode_reward / env.max_reward
            )

    avg_rewards_by_util = sorted(
        [
            (util, np.average(rewards))
            for util, rewards in rewards_by_utilization.items()
        ]
    )

    return Q_table, training_set, rewards_per_episode, state_count, avg_rewards_by_util


def q_learning_test(
    Q_table, training_set=None, n_tasksets=10000, return_schedules=False
):
    """
    Takes in a Q-matrix representing a QL scheduler to evaluate. If training_set is specified, uses it to compute success rate (particularly the rewards_by_utilization). If it's not specified, generates a random test_set of tasksets.
    Args:
        Q_table - the matrix of values to use to decide which actions to take in each state
        training_set - optional set to use for evaluation
        n_tasksets - if training_set not specified, the number of tasksets to create for the test_set
        return_schedules - boolean value to return the schedules for each taskset in training_set
    """
    rewards_by_utilization: DefaultDict[float, List[float]] = defaultdict(list)
    schedules_by_reward: DefaultDict[
        float, List[Tuple[List[Task], List[int]]]
    ] = defaultdict(list)

    if not training_set:
        training_set = [utils.generate_random_taskset() for _ in range(n_tasksets)]

    for tasks in training_set:
        env = TaskSchedulingEnvironment(tasks=tasks, n_quanta=5)
        current_state = tuple(sorted(env.state, reverse=True))
        total_episode_reward = 0
        schedule = []

        for i in range(env.n_iter_episode):
            if current_state not in Q_table:
                Q_table[current_state] = np.zeros(len(current_state))

            if env.no_jobs_to_do():
                action = IDLE_TASK_ID
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
            total_episode_reward = total_episode_reward + reward
            current_state = next_state
            if return_schedules:
                schedule.append(action)

        utilization = sum(task.exectime / task.period for task in tasks)

        reward = total_episode_reward / env.max_reward
        rewards_by_utilization[round(utilization * 20) / 20].append(reward)

        if return_schedules:
            schedules_by_reward[reward].append((tasks, schedule))

    avg_rewards_by_util = sorted(
        [
            (util, np.average(rewards))
            for util, rewards in rewards_by_utilization.items()
        ]
    )
    return avg_rewards_by_util, schedules_by_reward


print("==============TRAINING MODE==============")

print("===(Training first with a smaller training_set over 5 times)===")

(
    Q_table,
    training_set,
    rewards_per_episode,
    state_count,
    tr_rewards_by_utilization,
) = q_learning(print_status=True)

trained_rewards_by_utilization, _ = q_learning_test(Q_table, training_set)

print("===(Now training again with a unique training_set)===")

(
    Q_table_unique_train,
    training_set_unique,
    _,
    _,
    tr_rewards_by_utilization_unique,
) = q_learning(n_tasksets=20000, n_repeat=1, print_status=True)

trained_rewards_by_utilization_unique_set, _ = q_learning_test(
    Q_table_unique_train, training_set_unique
)


print("==============TESTING MODE==============")

test_rewards_by_utilization, schedules_by_reward = q_learning_test(
    Q_table, return_schedules=True
)

_, _, _, _, rand_rewards_by_utilization = q_learning(
    n_tasksets=5000, n_repeat=1, min_exploration_proba=1
)


print("==============PLOTTING==============")

fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
ax.set_title("Average rewards by utilization")
ax.plot(
    [u for u, _ in tr_rewards_by_utilization],
    [r for _, r in tr_rewards_by_utilization],
    label="During-training mode",
)
ax.plot(
    [u for u, _ in trained_rewards_by_utilization],
    [r for _, r in trained_rewards_by_utilization],
    label="Post-training mode (repeat training_set 5x)",
)
ax.plot(
    [u for u, _ in trained_rewards_by_utilization_unique_set],
    [r for _, r in trained_rewards_by_utilization_unique_set],
    label="Post-training mode (unique training_set)",
)
ax.plot(
    [u for u, _ in test_rewards_by_utilization],
    [r for _, r in test_rewards_by_utilization],
    label="Testing mode",
)
ax.plot(
    [u for u, _ in rand_rewards_by_utilization],
    [r for _, r in rand_rewards_by_utilization],
    label="Random scheduling mode",
)
ax.set_xlabel("utilization")
ax.set_ylabel("reward")
ax.legend()
plt.show()


print("==============EXAMPLE SCHEDULES==============")

i = 0
for reward, tasks_schedules in schedules_by_reward.items():
    i += 1
    print(f"\nTask set #{i}")

    tasks, schedule = tasks_schedules[np.random.choice(len(tasks_schedules))]
    print(f"utilization: {sum(task.exectime / task.period for task in tasks)}")
    print(f"reward: {reward}")

    utils.print_by_task(tasks, schedule)
