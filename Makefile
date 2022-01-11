format:
	# Sort
	python -m isort ./rtsrl

	# Format
	python -m black --target-version py37 ./rtsrl

lint:
	python -m mypy ./rtsrl
	python -m isort ./rtsrl --check-only
	python -m flake8 ./rtsrl
	python -m black --check ./rtsrl

run:
	python ./rtsrl/main.py
