
import numpy as onp
import jax
import os
import time

import cardiax
from nnfe import NNFE

cardiax.set_jax_enable_x64(False)

os.makedirs("results", exist_ok=True)

nnfe = NNFE.from_yaml("inputs/nnfe_params.yaml")

toc = time.time()
nnfe.train()
tic = time.time()
