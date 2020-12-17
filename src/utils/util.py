import os
import sys
import argparse
from pathlib import Path

import git
import yaml
from addict import Dict


def get_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config",
        default="./configs/sample.yaml",
        type=str, help="config path")
    arg("-k", "--k_fold_num",
        default=0, type=int,
        help="validation index for cross-validation")
    return parser


def read_yaml(path):
    try:
        with open(path) as file:
            obj = yaml.safe_load(file)
    except Exception as e:
        print('Exception occurred while loading YAML...', file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
    return Dict(obj)


def git_commits(my_func):
    def wrapper(*args, **kwargs):
        repo = git.Repo(str(Path(os.getcwd()).parents[0]))
        repo.git.diff("HEAD")
        repo.git.add(".")
        repo.index.commit("Auto Commit. (Before the experiment)")
        my_func(*args, **kwargs)
    return wrapper
