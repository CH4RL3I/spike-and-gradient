"""spike-and-gradient: linear regression, a biological neuron and a neural network in NumPy."""

import os

# Other jobs share the machine: keep BLAS thread pools small unless the caller overrides.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_var, "4")

__version__ = "0.1.0"
