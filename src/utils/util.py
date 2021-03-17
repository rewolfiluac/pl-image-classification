import os
from pathlib import Path

import git


def git_commits(my_func):
    def wrapper(*args, **kwargs):
        repo = git.Repo(str(Path(os.getcwd()).parents[0]))
        repo.git.diff("HEAD")
        repo.git.add(".")
        repo.index.commit("Auto Commit. (Before the experiment)")
        my_func(*args, **kwargs)

    return wrapper
