import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")


def _get_project_dir(location):
    """Helper function for creating project config paths.

    Args:
        location: Directory inside PROJECT_ROOT/configs/ to create path for

    Returns:
        String giving path of PROJECT_ROOT/configs/location
    """
    return os.path.join(CONFIG_DIR, location)


EXPERIMENT_CONFIG_DIR = _get_project_dir("experiment")
MODEL_CONFIG_DIR = _get_project_dir("model")
DATA_CONFIG_DIR = _get_project_dir("data")
