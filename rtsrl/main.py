import utils
from schedulers import randomly_schedule, rate_monotonic_schedule


def main():

    # filename = input("What file contains the tasks to schedule?\n")
    filename = "data/taskset2.txt"

    tasks = utils.read_tasks(filename)
    lcm_period = utils.get_lcm_period(tasks)
    utils.assert_is_schedulable(tasks)

    print("\n===========================================\n")
    utils.print_tasks(tasks)

    print("\n============RANDOM SCHEDULING=============\n")
    schedule = randomly_schedule(tasks=tasks, runtime=lcm_period)
    utils.print_by_task(tasks, schedule, print_descrip=True)

    print("\n=========RATE-MONOTONIC SCHEDULING==========\n")
    schedule = rate_monotonic_schedule(tasks=tasks, runtime=lcm_period)
    utils.print_by_task(tasks, schedule, print_descrip=True)

    # testing utils.generate_random_taskset()
    # for i in range(5):
    #     tasks = utils.generate_random_taskset()
    #     utilization = sum(task.exectime / task.period for task in tasks)
    #     utils.assert_under_constraints(tasks, max_utilization=1, max_exectime=20)
    #     print(f"num_tasks: {len(tasks)}\nutilization: {utilization}")
    #     print(f"TASKS: {tasks}")


if __name__ == "__main__":
    main()
