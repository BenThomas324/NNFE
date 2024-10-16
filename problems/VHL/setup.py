
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
