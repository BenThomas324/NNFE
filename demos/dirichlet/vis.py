
import jax.numpy as np
import jax
import time
import matplotlib.pyplot as plt

from nnfe.control.natural import NNFE
from cardiax.solvers.newton import Newton_Solver
import pyvista as pv

pv.OFF_SCREEN = True

nnfe = NNFE("test/73966/input_files/test_params.yaml")

nnfe.problem = nnfe.fe_handler.problem

timing = False
make_fig = False
plot_error = True
make_movie = False

X = nnfe.sampler.X

if timing:
    toc = time.time()
    nn_sols = jax.vmap(nnfe.evaluate)(X)
    jitting_time = time.time() - toc

    toc = time.time()
    nn_sols = jax.vmap(nnfe.evaluate)(X)
    jitted_time = time.time() - toc

if make_fig:
    solver = Newton_Solver(nnfe.problem, np.zeros((nnfe.problem.num_total_dofs_all_vars,)))
    temp_vars = nnfe.problem.internal_vars_surfaces["u"]["bc2"]["t"]
    nnfe.problem.set_internal_vars_surfaces({"u": {"bc2": {"t": X[-1] * temp_vars}}})
    sol, info = solver.solve(max_iter=40)

    pvmesh = pv.from_meshio(nnfe.problem.mesh["u"])

    # Create a plotter with off-screen rendering
    nn_sol = nnfe.evaluate(X[-1])
    pl = pv.Plotter(off_screen=True)
    pvmesh.point_data["nn_sol"] = np.array(nn_sol).reshape(-1, 3)
    pvmesh.point_data["fe_sol"] = np.array(sol).reshape(-1, 3)
    warped_fe = pvmesh.warp_by_vector("fe_sol", factor=1.0)
    warped_nn = pvmesh.warp_by_vector("nn_sol", factor=1.0)

    pl.add_mesh(warped_fe, opacity=0.8, color="gainsboro", show_edges=True, scalars=None)
    pl.add_mesh(warped_nn, style="points", render_points_as_spheres=True, point_size=5, color="red")
    pl.screenshot('test.png')

if plot_error:

    for title, datapts in (["X", "Y"], [nnfe.sampler.X, nnfe.sampler.Y]):
    # datapts = np.linspace(0, 0.5, 11)[:, None]
        init_sol = np.zeros((nnfe.problem.num_total_dofs_all_vars))
        solver = Newton_Solver(nnfe.problem, init_sol, line_search_flag=True)
        temp_vars = np.ones_like(nnfe.problem.internal_vars_surfaces["u"]["bc2"]["t"])

        nn_sols = jax.vmap(nnfe.evaluate)(datapts)
        fe_sols = []
        for i, pt in enumerate(datapts):
            print("Traction value of ", pt)
            solver.initial_guess = nn_sols[i]
            nnfe.problem.set_internal_vars_surfaces({"u": {"bc2": {"t": pt * temp_vars}}})
            sol, info = solver.solve(max_iter=40)
            assert info[0]
            fe_sols.append(np.array(sol).reshape(-1, 3))

        fe_sols = np.array(fe_sols)
        nn_sols = np.array(nn_sols).reshape(fe_sols.shape[0], -1, 3)
        diffs = np.linalg.norm(fe_sols - nn_sols, axis=-1)
        mean_diffs = np.mean(diffs, axis=-1)
        max_diffs = np.max(diffs, axis=-1)

        fig, ax = plt.subplots()
        ax.plot(datapts, mean_diffs, label="Mean error")
        ax.plot(datapts, max_diffs, label="Max error")
        ax.set_xlabel("Data Points")
        ax.set_ylabel("Error")
        ax.legend()
        plt.savefig(f"error_plot_{title}.png")

if make_movie:
    datapts = np.linspace(0, 1, 11)[:, None]
    sol = np.zeros((nnfe.problem.num_total_dofs_all_vars))
    solver = Newton_Solver(nnfe.problem, sol, line_search_flag=True)
    temp_vars = np.ones_like(nnfe.problem.internal_vars_surfaces["u"]["bc2"]["t"])

    for i, pt in enumerate(datapts):
        print("Traction value of ", pt)
        solver.initial_guess = sol
        nnfe.problem.set_internal_vars_surfaces({"u": {"bc2": {"t": pt * temp_vars}}})
        sol, info = solver.solve(max_iter=40)
        assert info[0]

        pvmesh = pv.from_meshio(nnfe.problem.mesh["u"])
        pl = pv.Plotter(off_screen=True)
        nn_sol = nnfe.evaluate(pt)
        pvmesh.point_data["nn_sol"] = np.array(nn_sol).reshape(-1, 3)
        pvmesh.point_data["fe_sol"] = np.array(sol).reshape(-1, 3)
        warped_fe = pvmesh.warp_by_vector("fe_sol", factor=1.0)
        warped_nn = pvmesh.warp_by_vector("nn_sol", factor=1.0)

        pl.add_mesh(warped_fe, opacity=0.8, color="gainsboro", show_edges=True, scalars=None)
        pl.add_mesh(warped_nn, style="points", render_points_as_spheres=True, point_size=5, color="red")
        filename = f"frame_{i:03d}.png"
        pl.close()

