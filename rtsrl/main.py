import utils
from schedulers import randomly_schedule


def main():
    print("Hello World!")

    # filename = input("What file contains the tasks to schedule?\n")
    filename = "data/taskset1.txt"

    tasks = utils.read_tasks(filename)
    lcm_period = utils.get_lcm_period(tasks)

    schedule = randomly_schedule(tasks=tasks, runtime=lcm_period)

    utils.print_tasks(tasks)
    utils.print_schedule(schedule)
    utils.print_by_task(tasks, schedule)
    utils.print_stats(schedule)


if __name__ == "__main__":
    main()
