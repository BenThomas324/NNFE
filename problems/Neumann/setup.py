
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
