
from cardiax import rectangle_mesh
import jax.numpy as np
import numpy as onp
import jax

mesh = rectangle_mesh(25, 25, 1., 1.)

def left(point):
    return np.isclose(0., point[0], atol=1e-5)

def right(point):
    return np.isclose(1., point[0], atol=1e-5)

def bottom(point):
    return np.isclose(0., point[1], atol=1e-5)

def top(point):
    return np.isclose(1., point[1], atol=1e-5)

for func in [left, right, bottom, top]:
    mask = jax.vmap(func)(mesh.points)
    mesh.point_data[func.__name__] = onp.array(mask).astype(onp.float32)

mesh.write("rect.vtk")