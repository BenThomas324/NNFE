
import jax.numpy as np
import meshio
import sys
from .FE_helpers import *
import functools as ft

from .env_var import path
sys.path.append(path)

from jax_fem.problem import Problem
from jax_fem.solver import apply_bc_vec
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, box_mesh

def VHL_setup(params, parent):

    print("Reading meshes")
    mesh_file = parent + params["mesh_file"]
    fiber_file = parent + params["fiber_file"]
    domain = meshio.read(mesh_file)
    fiber_mesh = meshio.read(fiber_file)
    ele_type = "TET4"
    cell_type = get_meshio_cell_type(ele_type)
    mesh = Mesh(domain.points, domain.cells_dict[cell_type])

    top_points = domain.points[domain.point_data["top_nodes"].astype(bool)]
    LV_points = domain.points[domain.point_data["LVendo_nodes"].astype(bool)]
    RV_points = domain.points[domain.point_data["RVendo_nodes"].astype(bool)]

    def top(point):
        return np.isclose(point, top_points, atol=1e-6).all(axis=1).any()

    def LVendo(point):
        return np.isclose(point, LV_points, atol=1e-6).all(axis=1).any()

    def RVendo(point):
        return np.isclose(point, RV_points, atol=1e-6).all(axis=1).any()

    # Fixed top nodes
    def zero_dirichlet(point):
        return 0.0

    # dirichlet_bc_info = [Where, what directions, value]
    dirichlet_bc_info = [[top]*3, [0, 1, 2], [zero_dirichlet]*3]
    location_fns = [LVendo, RVendo]

    fibers = fiber_mesh.cell_data["fiber"][0][:, None, :]
    sheets = fiber_mesh.cell_data["sheet"][0][:, None, :]
    normals = fiber_mesh.cell_data["normal"][0][:, None, :]

    print("initalize problem")
    problem = VHL_Fung(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)

    LV_bdry = problem.fes[0].get_boundary_conditions_inds([LVendo])[0]
    RV_bdry = problem.fes[0].get_boundary_conditions_inds([RVendo])[0]

    LV_normals = get_normals(problem.fes[0], LV_bdry)
    RV_normals = get_normals(problem.fes[0], RV_bdry)

    problem.set_params(params["constants"], (25, 5), 5, [LV_normals[:, None, :], RV_normals[:, None, :]], [fibers, sheets, normals])

    return problem, [fibers, sheets, normals], [LV_normals, RV_normals]

def PS_test(params):

    print("Reading meshes")
    domain = meshio.read("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/LV/mesh/LV_complete.vtu")
    ele_type = "TET4"
    cell_type = get_meshio_cell_type(ele_type)
    mesh = Mesh(domain.points, domain.cells_dict[cell_type])

    base_points = domain.points[domain.point_data["BASE"].astype(bool)[:, 0]]
    LV_points = domain.points[domain.point_data["ENDO"].astype(bool)[:, 0]]

    def base(point):
        return np.isclose(point, base_points, atol=1e-6).all(axis=1).any()

    def LVendo(point):
        return np.isclose(point, LV_points, atol=1e-6).all(axis=1).any()

    # Fixed top nodes
    def zero_dirichlet(point):
        return 0.0

    # dirichlet_bc_info = [Where, what directions, value]
    dirichlet_bc_info = [[base]*3, [0, 1, 2], [zero_dirichlet]*3]
    location_fns = [LVendo]

    fibers = domain.point_data["fibers"]
    sheets = domain.point_data["sheets"]
    normals = domain.point_data["normals"]

    print("initalize problem")
    problem = LV_Fung(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)

    LV_bdry = problem.fes[0].get_boundary_conditions_inds([LVendo])[0]
    LV_normals = get_normals(problem.fes[0], LV_bdry)
    
    fibers = problem.fes[0].convert_from_dof_to_quad(fibers)
    sheets = problem.fes[0].convert_from_dof_to_quad(sheets)
    normals = problem.fes[0].convert_from_dof_to_quad(normals)

    problem.set_params(params["constants"], 1., 0., LV_normals[:, None, :], [fibers, sheets, normals])

    return problem, [fibers, sheets, normals], LV_normals

def LV_test(params):

    print("Reading meshes")
    domain = meshio.read("/home/bthomas/Desktop/Research/JAXFEM_temp/NNFE/nnfe-scratch-rewrite/LV/mesh/LV_complete.vtu")
    ele_type = "TET4"
    cell_type = get_meshio_cell_type(ele_type)
    mesh = Mesh(domain.points, domain.cells_dict[cell_type])

    base_points = domain.points[domain.point_data["BASE"].astype(bool)[:, 0]]
    LV_points = domain.points[domain.point_data["ENDO"].astype(bool)[:, 0]]

    def base(point):
        return np.isclose(point, base_points, atol=1e-6).all(axis=1).any()

    def LVendo(point):
        return np.isclose(point, LV_points, atol=1e-6).all(axis=1).any()

    # Fixed top nodes
    def zero_dirichlet(point):
        return 0.0

    # dirichlet_bc_info = [Where, what directions, value]
    dirichlet_bc_info = [[base]*3, [0, 1, 2], [zero_dirichlet]*3]
    location_fns = [LVendo]

    fibers = domain.point_data["fibers"]
    sheets = domain.point_data["sheets"]
    normals = domain.point_data["normals"]

    print("initalize problem")
    problem = LV_Fung(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)

    LV_bdry = problem.fes[0].get_boundary_conditions_inds([LVendo])[0]
    LV_normals = get_normals(problem.fes[0], LV_bdry)
    
    # Plot normals
    # cell_points = onp.take(problem.fes[0].points, problem.fes[0].cells, axis=0)
    # cell_face_points = onp.take(cell_points, problem.fes[0].face_inds, axis=1)
    # normals = onp.zeros_like(cell_points)
    # normals[LV_bdry[:, 0], LV_bdry[:, 1]] = LV_normals[:, 0, 0, :]
    
    # fct = lambda pt: (pt[0]**2 + pt[1]**2 + pt[2]**2)**(.5)
    # vals = jax.vmap(jax.vmap(fct))(cell_points)

    # domain.cell_data["arrows"] = normals.sum(axis=1)
    # domain.write("Test.vtu")

    # domain2 = meshio.read("Test.vtu")

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # selected_coos = cell_points[LV_bdry[:, 0], LV_bdry[:, 1]]
    # ax.scatter(*selected_coos.T)
    # ax.quiver(*selected_coos.T, *LV_normals[:, 0, 0, :].T)
    # plt.savefig("Test.png")

    fibers = problem.fes[0].convert_from_dof_to_quad(fibers)
    sheets = problem.fes[0].convert_from_dof_to_quad(sheets)
    normals = problem.fes[0].convert_from_dof_to_quad(normals)

    problem.set_params(params["constants"], 1., 0., LV_normals[:, None, :], [fibers, sheets, normals])

    # from jax_fem.solver import solver
    # from jax_fem.utils import save_sol
    # sol = solver(problem, line_search_flag=True)
    # save_sol(problem.fes[0], sol[0], "vtus/sol_LVNH_TCa.vtu")
    # exit()

    return problem, [fibers, sheets, normals], LV_normals


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
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

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

def Neumann_test():
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
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    # Define Dirichlet boundary values.
    def zero_bc(point):
        return 0.

    dirichlet_bc_info = [[left] * 3, [0, 1, 2],
                        [zero_bc] * 3]

    problem = Pressure_cube(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=[right])

    right_inds = problem.fes[0].get_boundary_conditions_inds([right])
    right_normals = get_normals(problem.fes[0], right_inds[0])
    
    # Solve to check range
    # # right_normals = problem.fes[0].face_normals[right_inds[0][:, 1]]
    # pressures = 3. * np.ones_like(right_normals)[:, :, :, :1]

    # problem.set_params(right_normals, pressures)
    # from jax_fem.solver import solver
    # from jax_fem.utils import save_sol

    # sol = solver(problem, line_search_flag=True)
    # save_sol(problem.fes[0], sol[0], "image/Pressure_test.vtu")
    # exit()
    return problem, right_normals

def get_training_points(params):
    PV_loop_list = onp.loadtxt(params["PV_file"])
    TCa_list = onp.loadtxt(params["T_Ca_file"])

    X = np.hstack((PV_loop_list[::4][:, :2], TCa_list[::4][:, None]))
    # p_min, p_max = PV_loop_list[:, 2].min(), PV_loop_list[:, 2].max()
    # T_Ca_min, T_Ca_max = TCa_list.min(), TCa_list.max()

    # p_range = np.linspace(p_min, p_max, 5)
    # T_Ca_range = np.linspace(T_Ca_min, T_Ca_max, 5)
    # p_grid, T_Ca_grid = np.meshgrid(p_range, T_Ca_range)
    # X = np.vstack((p_grid.reshape(-1), 0.2 * p_grid.reshape(-1), T_Ca_grid.reshape(-1))).T
    return X
