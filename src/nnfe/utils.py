"""
This file is meant to handle all aspects of handling the filesystem.
It will create directories, save models, and load files.
(May want to rename)
"""

import yaml
import numpy as onp
from pathlib import Path

class Utilities:

    def __init__(self, utility_params):
        
        if utility_params["save"]:
            parent = Path(utility_params["parent_dir"]) / Path(utility_params["name"])

            temp_key = onp.random.randint(1e5)
            while (parent / f"/{temp_key}").exists():
                temp_key = onp.random.randint(1e5)
            parent = parent / Path(f"{temp_key}")

            self.key = temp_key
            self.dirs_params = utility_params["extra_dirs"]
            self.output_params = utility_params["output"]

            for dir_path in self.dirs_params.values():
                dir_path = parent / dir_path
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)

            if self.output_params["print"]:
                self.print = self.output_params["print"]
            else:
                self.print = False

            if self.output_params["saveat"]:
                self.save = self.output_params["saveat"]
            else:
                self.save = False

            onp.savetxt(parent + "/running.txt", onp.array([0]))

        else:
            # Figure out what else to do here...
            self.print = False
            self.save = False
            self.key = None
            self.dirs_params = {}
            self.output_params = {}

        return

def create_dirs(params, results_dir):
    # Currently done via RNG, but should change to user specified file
    # and throw error if it already exists
    temp_key = onp.random.randint(1e5)
    while (results_dir / f"/{temp_key}").exists():
        temp_key = onp.random.randint(1e5)

    results_dir += f"/{temp_key}"
    (results_dir).mkdir()
    (results_dir + "/plots").mkdir()
    (results_dir + "/values").mkdir()
    
    with open(results_dir / "/params.yaml", "w") as f:
        yaml.dump(params, f)

    onp.savetxt(results_dir + "/running.txt", onp.array([0]))

    return results_dir, temp_key

