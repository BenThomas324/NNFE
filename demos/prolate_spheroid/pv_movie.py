
import pyvista as pv
import numpy as np
import glob
import matplotlib.pyplot as plt

parent = "Results/87285/results"

for j in [0, 1, 2]:
    print("Creating movie for loop ", j+1)
    file_dir = f"{parent}/pv_loop_{j+1}/u_*.vtk"
    files = sorted(glob.glob(file_dir))
    mesh = pv.read(files[0])

    disp_mean = []
    disp_max = []

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, show_edges=True)
    plotter.zoom_camera(1.25)
    plotter.open_gif(f"{parent}/pv_loop_{j+1}.gif", fps=5)
    for i, file in enumerate(files):
        mesh = pv.read(file)
        warped_fe = mesh.warp_by_vector("fe_sol", factor=1.0)
        warped_nn = mesh.warp_by_vector("nn_sol", factor=1.0)
        plotter.add_mesh(warped_fe, show_edges=True, scalars=None)
        plotter.add_mesh(warped_nn, style="points", render_points_as_spheres=True, point_size=10, color="red")
        plotter.add_text(f"Time: {i}", position="lower_edge")
        plotter.write_frame()
        plotter.clear_actors()

        diff = np.linalg.norm(mesh["nn_sol"].reshape(-1, 3) \
                            - mesh["fe_sol"].reshape(-1, 3), axis=1)
        disp_mean.append(diff.mean())
        disp_max.append(diff.max())

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(disp_mean)
    ax[0].set_title("Mean displacement error")
    ax[0].set_xlabel("Time step")
    ax[0].set_ylabel("Error (cm)")
    ax[1].plot(disp_max)
    ax[1].set_title("Max displacement error")
    ax[1].set_xlabel("Time step")
    ax[1].set_ylabel("Error (cm)")
    plt.tight_layout()
    plt.savefig(f"{parent}/pv_loop_{j+1}_error.png")

    plotter.close()

