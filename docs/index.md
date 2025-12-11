
# Getting Started with CARDIAX_NNFE

## What is NNFE?

NNFE stands for Neural Network Finite Element which is a method used for scientific machine learning. The NNFE method is designed to learn the parameter-to-solution mapping that is defined by the residual of the PDE. We are trying to build a robust representation of the PDE operator for fast evaluations. Below describes how to install the software, the dependencies we rely on, and links to examples.

## Dependencies

The major dependencies for CARDIAX-NNFE are:

- JAX: Google's high performance computing package based on GPUs for vectorization and parallelization
- CARDIAX: Finite Element library built on JAX
- EQUINOX: Neural Network utility library for making custom networks and general extensions of JAX
- OPTAX: Optimization library for machine learning

## Installation

After installing the dependecies, CARDIAX-NNFE is installed as an active environment. Navigate to the directory `../NNFE` then run
```
pip install -e .
```
This will install NNFE locally through pip and allow editing of the library. Further development will release a true package to pip install from.


