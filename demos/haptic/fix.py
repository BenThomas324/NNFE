
import meshio
import jax.numpy as np
import numpy as onp
import jax

mesh = meshio.read("msh/box.vtk")
mesh.points = onp.array(mesh.points).astype(onp.float32)

left_pin = np.array([0., .5, .5])
right_pin = np.array([1., .5, .5])

def find_left(point):
    m1 = np.linalg.norm(point - left_pin) <= 0.21
    m2 = np.isclose(point[0], 0.0)
    return np.logical_and(m1, m2)

def find_right(point):
    m1 = np.linalg.norm(point - right_pin) <= 0.21
    m2 = np.isclose(point[0], 1.0)
    return np.logical_and(m1, m2)

left_mask = jax.vmap(find_left)(mesh.points)
right_mask = jax.vmap(find_right)(mesh.points)
mesh.point_data["left_pin"] = onp.array(left_mask).astype(onp.float32)
mesh.point_data["right_pin"] = onp.array(right_mask).astype(onp.float32)

mesh.write("msh/box_pins.vtk")

print()

