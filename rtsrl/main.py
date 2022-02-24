import utils
from schedulers import rate_monotonic_schedule


def main():

    # filename = input("What file contains the tasks to schedule?\n")
    filename = "data/taskset2.txt"

    tasks = utils.read_tasks(filename)
    lcm_period = utils.get_lcm_period(tasks)
    utils.assert_is_schedulable(tasks)

    # schedule = randomly_schedule(tasks=tasks, runtime=lcm_period)
    schedule = rate_monotonic_schedule(tasks=tasks, runtime=lcm_period)

    print("\n===========================================\n")
    utils.print_tasks(tasks)
    print("\n===========================================\n")
    utils.print_schedule(schedule)
    print("\n===========================================\n")
    utils.print_by_task(tasks, schedule)
    print("\n===========================================\n")
    utils.print_stats(schedule)


if __name__ == "__main__":
    main()
