
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

