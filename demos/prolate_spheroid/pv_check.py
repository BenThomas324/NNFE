
import numpy as onp
import jax
import equinox as eqx
import time
import os

from nnfe import NNFE
from cardiax import Newton_Solver

import argparse

if __name__ == "__main__":
    # allow for batch-processing of NN runs using CLI + argparse
    parser = argparse.ArgumentParser(
                    prog='pv check; postprocessing tool',
                    description='Post-processes a trained PV Loop',
                    epilog='Please contact us with any questions, comments, or concerns!')

    # access trained network parameters saved in the given run number
    parser.add_argument('-r', '--run_num', type=str, default=None)   
    args = parser.parse_args()
    run_num = args.run_num

    parent = "Results/" + run_num + "/"
    nnfe = NNFE.from_yaml(parent + "config_files/resolved_nnfe_config.yaml")
    nnfe.ml.network = eqx.tree_deserialise_leaves(parent + "models/model.eqx", nnfe.ml.network)

    solver = Newton_Solver(nnfe.problem, onp.zeros((nnfe.problem.num_total_dofs_all_vars)))

    pv_error = []
    pv_times = []
    for j in [1, 2, 3]:
        jit_times = []
        os.makedirs(parent + f"results/pv_loop_{j}", exist_ok=True)
        data = onp.genfromtxt(f"PV_loops/tp_loop_{j}.csv", delimiter=",", skip_header=1)
        data[:, [0, 1]] = data[:, [1, 0]]

        # Time PV loop
        timed_eval = eqx.filter_jit(nnfe.evaluate)

        t = time.time()
        nn_sols = jax.vmap(timed_eval)(data)
        jit_times.append(time.time() - t)

        t = time.time()
        nn_sols = jax.vmap(timed_eval)(data)
        jit_times.append(time.time() - t)

        fe_sols = []
        stress_mean = []
        stress_std = []
        stress_max = []
        max_vals = []
        for i, x in enumerate(data):
            int_vars = nnfe.nnfe_set_int_vars(x)
            int_vars_surfaces = nnfe.nnfe_set_int_vars_surf(x)
            solver.initial_guess = nn_sols[i]
            sol, info = solver.solve(max_iter=25)
            assert info[0]

            fe_sols.append(sol)

        pv_array = onp.zeros((len(data), 2))
        for i in range(len(data)):
            diff_disps = onp.linalg.norm(nn_sols[i].reshape(-1, 3) - fe_sols[i].reshape(-1, 3), axis=1)
            pv_array[i, 0] = diff_disps.mean()
            pv_array[i, 1] = diff_disps.max()

            nnfe.problem.mesh["u"].point_data["nn_sol"] = onp.array(nn_sols[i]).reshape(-1, 3)
            nnfe.problem.mesh["u"].point_data["fe_sol"] = onp.array(fe_sols[i]).reshape(-1, 3)
            nnfe.problem.mesh["u"].save(parent + f"results/pv_loop_{j}/u_{i:02}.vtk")
        
        pv_error.append(pv_array)
        pv_times.append(onp.array(jit_times))

    for j in [0, 1, 2]:
        print(pv_times[j])
        print("Mean L2 error:   ", pv_error[j][:, 0].mean())
        print("Mean Linf error: ", pv_error[j][:, 1].mean())