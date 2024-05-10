import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")


def _get_project_dir(location):
    return os.path.join(CONFIG_DIR, location)


EXPERIMENT_CONFIG_DIR = _get_project_dir("experiment")
MODEL_CONFIG_DIR = _get_project_dir("model")
DATA_CONFIG_DIR = _get_project_dir("data")
