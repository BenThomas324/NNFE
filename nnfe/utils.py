
import os
import yaml
import numpy as onp

def setup_dirs(params, results_dir):
    temp_key = onp.random.randint(1e5)
    while os.path.exists(results_dir + f"/{temp_key}"):
        temp_key = onp.random.randint(1e5)

    results_dir += f"/{temp_key}"
    os.makedirs(results_dir)
    os.makedirs(results_dir + "/plots")
    os.makedirs(results_dir + "/values")
    
    with open(results_dir + "/params.yaml", "w") as f:
        yaml.dump(params, f)

    onp.savetxt(results_dir + "/running.txt", onp.array([0]))

    return results_dir, temp_key

def prob_setup(prob_dir):
    
    return





