
import pyvista as pv
import argparse
import glob

pv.start_xvfb()
pv.OFF_SCREEN = True

parser = argparse.ArgumentParser(
                    prog='Visualizer',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('filename')
args = parser.parse_args()
result_dir = f"/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/results/VHL/{args.filename}"

NN_files = sorted(glob.glob(result_dir + "/NN/*.vtu"))
# FE_files = sorted(glob.glob(result_dir + "/FE/*.vtu"))

for i in range(len(NN_files)):
    NN_mesh = pv.read(NN_files[i])
    # FE_mesh = pv.read(FE_files[-1])

    NN_mesh.set_active_vector = "sol"
    # FE_mesh.set_active_vector = "sol"
    NN_grid = NN_mesh.warp_by_vector()
    # FE_grid = FE_mesh.warp_by_vector()

    p_NN = pv.Plotter()
    p_NN.add_mesh(NN_grid)
    p_NN.screenshot(result_dir + f"/NN_{i:02d}.png")

    # p_FE = pv.Plotter()
    # p_FE.add_mesh(FE_grid)
    # p_FE.screenshot(result_dir + "/FE.png")
