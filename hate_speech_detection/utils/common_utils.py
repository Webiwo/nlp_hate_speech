import yaml
import os


def read_yaml(path_to_yaml: str) -> dict:
    """Read YAML file and return content as dictionary."""
    with open(path_to_yaml, "r") as file:
        return yaml.safe_load(file)


def create_directories(paths: list):
    """Create directories from a list of paths."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
