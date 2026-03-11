
# Introduction to CARDIAX_NNFE

CARDIAX_NNFE provides a framework for finding the parameterized solution of a PDE. We will showcase an example for a hyperelastic cube where the value of pressure is varied. As the name suggests, there are two major components to the NNFE method: the traditional Finite Element framework, and the neural network functionality. The code utilizes the NNFE controller to keep track of everything that is going on. The NNFE controller has 5 submodules, two major and three minor. The two major submodules are finite element controller and the machine learning controller. The three minor submodules are a sampler, plotter, and utility assistance. 

Place figure here to help explain

## Finite Element Setup

The finite element controller is built off the input files used in CARDIAX.
