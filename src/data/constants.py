import os

import git

_repository_path = str(
    git.Repo(__file__, search_parent_directories=True).working_tree_dir
)
DATA_PATH = os.path.join(os.path.join(_repository_path, "src"), "data")
CONFIG_PATH = os.path.join(DATA_PATH, "config")

DATASET_PATH = os.path.join(_repository_path, "data")
FINANCE_DATA_PATH = os.path.join(DATASET_PATH, "finance")
QRT_DATA_PATH = os.path.join(DATASET_PATH, "qrt")
