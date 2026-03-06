
import numpy as onp
import jax
import equinox as eqx
import os

from nnfe import NNFE
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

jax.config.update("jax_enable_x64", False)
    
nnfe = NNFE.from_yaml("configs/nnfe_params.yaml")

nnfe.train()
