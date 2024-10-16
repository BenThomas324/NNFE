
from jax_fem.generate_mesh import get_meshio_cell_type, box_mesh, Mesh
import numpy as onp

def Dirichlet_test():
    # Specify mesh-related information (first-order hexahedron element).
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly, Lz = 1., 1., 1.
    meshio_mesh = box_mesh(Nx=10,
                        Ny=10,
                        Nz=10,
                        Lx=Lx,
                        Ly=Ly,
                        Lz=Lz,
                        data_dir="mesh",
                        ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # Define boundary locations.
    def left(point):
        return onp.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return onp.isclose(point[0], Lx, atol=1e-5)

    # Define Dirichlet boundary values.
    def zero_bc(point):
        return 0.

    def dirichlet_val_x2(amp, point):
        return amp * (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
                (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1])


    def dirichlet_val_x3(amp, point):
        return amp * (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
                (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2])


    dirichlet_bc_info = [[left] * 3 + [right] * 3, [0, 1, 2] * 2,
                        [zero_bc, dirichlet_val_x2, dirichlet_val_x3] +
                        [zero_bc] * 3]

    def get_bcs(amp):
        return [[left] * 3 + [right] * 3, [0, 1, 2] * 2,
                        [zero_bc, ft.partial(dirichlet_val_x2, amp), 
                         ft.partial(dirichlet_val_x3, amp)] +
                        [zero_bc] * 3]

    problem = NH_cube(mesh,
                          vec=3,
                          dim=3,
                          ele_type=ele_type)

    return problem, get_bcs
