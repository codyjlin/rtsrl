from setuptools import setup

NAME = "rtsrl"
DESCRIPTION = "Creating a real-time scheduler via reinforcement learning"
AUTHOR = "Cody Lin"
REQUIRES_PYTHON = ">=3.7.0"

REQUIRED = [
    "flake8",
    "black==21.10b0",
    "isort>=5",
    "mypy==0.790",
]

setup(
    name=NAME,
    author=AUTHOR,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    # https://stackoverflow.com/questions/28509965/setuptools-development-requirements
    # Install dev requirements with: pip install -e .
)
