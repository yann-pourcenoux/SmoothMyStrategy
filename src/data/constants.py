"""Module to hold some constants useful for the data related modules."""

import os

import git

_repository_path = str(git.Repo(__file__, search_parent_directories=True).working_tree_dir)
DATA_PATH = os.path.join(os.path.join(_repository_path, "src"), "data")
DATA_CONFIG_PATH = os.path.join(DATA_PATH, "cfg")

DATASET_PATH = os.path.join(_repository_path, "data")
