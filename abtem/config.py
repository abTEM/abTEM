import numpy as np
import cupy as cp

DTYPE = np.float32
CUPY_DTYPE = cp.float32
FFTW_THREADS = 16

if DTYPE == np.float32:
    COMPLEX_DTYPE = np.complex64
elif DTYPE == np.float64:
    COMPLEX_DTYPE = np.complex128
else:
    raise RuntimeError('')

if CUPY_DTYPE == cp.float32:
    COMPLEX_DTYPE = cp.complex64
else:
    raise RuntimeError('')