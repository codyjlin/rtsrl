# Creating a Real-time Scheduler via Reinforcement Learning

This is my [CPSC 490 Senior Project](https://dus.cs.yale.edu/490.html) for Spring 2022. The goal is to 1) use Reinforcement Learning to 2) build a scheduler that can ’schedule’ periodic jobs. There exist known schedulers like EDF (earliest deadline first) and RM (rate-monotonic), but we want to LEARN a new scheduler that makes jobs meet their deadlines.

This project will include at least the following parts:

1. a simple simulator
2. a random scheduling algorithm
3. a RL-based scheduling algorithm


## Tutorial

1. First clone this repo onto your local machine:

	```bash
	git clone https://github.com/codyjlin/rtsrl.git
	```

2. Then switch into the directory:

	```bash
	cd rtsrl
	```

3. To run the main Python program, run the command:

	```bash
	make run
	```

	:warning: You may need to follow step 1 of the **Environment Setup** section below.

4. To run the current q-learning script (WIP), run the command:

	```bash
	make run-q-learning
	```


## Environment Setup

1. Ensure that you have `make`, `python` (3.7 and above) with `pip`, installed. If your default `python` is not `python3`, a simple solution is to use `alias` in the terminal:
    
    ```bash
	alias python=python3
	alias pip=pip3
	```

2. [*Recommended*] Create a virtual environment to isolate development dependencies to be installed:
	
	```bash
	python -m venv venv/
	source venv/bin/activate
	echo $VIRTUAL_ENV  # should print path to venv
	```

3. Upgrade pip if outdated:
	
	```bash
	pip install --upgrade pip
	```

4. Install development dependencies:
	
	```bash
	pip install -e .
	```

## Code Style & Linting

The Python code in this repo:

- Conforms to [`Black` code style](https://black.readthedocs.io/en/stable/)
- Has type annotations as enforced by [`mypy`](https://mypy.readthedocs.io/en/stable/introduction.html)
- Has imports sorted by [`isort`](https://pycqa.github.io/isort/)
- Is lintable by [`flake8`](https://flake8.pycqa.org/en/latest/)

To ensure your Python code conforms to the repo standards:

1. Autoformat your code to conform to the code style:

	```bash
	make format
	```

2. Lint your code before submitting it for review:

	```bash
	make lint
	```
