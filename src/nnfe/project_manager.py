"""
This file is meant to handle all aspects of handling the filesystem.
It will create directories, save models, and load files.
(May want to rename)
"""

import numpy as onp
from pathlib import Path

class ProjectManager:

    def __init__(self, config):

        self.config = config
        self.save = config.save if config.save is not None else False

        if config.save:
            self.parent, self.key = self.generate_parent_and_rng()
            self.parent.mkdir(parents=True, exist_ok=False)
            self.paths = self.create_dirs()
        else:
            self.parent = None
            self.key = None
            self.paths = {}

        self.print = config.print_progress if config.print_progress is not None else False
        self.save_progress = config.save_progress if config.save_progress is not None else False

        if config.trained_weights_path is not None:
            self.trained_weights_path = config.trained_weights_path

        return

    def generate_parent_and_rng(self):
        parent = Path(self.config.parent_dir) / Path(self.config.name)

        temp_key = onp.random.randint(1e5)
        while (parent / f"{temp_key}").exists():
            temp_key = onp.random.randint(1e5)
        return parent / Path(f"{temp_key}"), temp_key

    def create_dirs(self):
        # Currently done via RNG, but should change to user specified file
        # and throw error if it already exists
        paths = {}
        for dir_key, dir_path in self.config.extra_dirs.items():
            dir_path = self.parent / dir_path
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
            paths[dir_key] = dir_path
        onp.savetxt(self.parent / "running.txt", onp.array([1]))
        return paths

