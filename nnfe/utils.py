"""
This file is meant to handle all aspects of handling the filesystem.
It will create directories, save models, and load files.
(May want to rename)
"""

import yaml
import numpy as onp
from pathlib import Path

def read_yaml(file):
    with open(file, "r") as f:
        return yaml.safe_load(f)

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

