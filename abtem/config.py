import numpy as np

DTYPE = np.float32
FFTW_THREADS = 16

if DTYPE == np.float32:
    COMPLEX_DTYPE = np.complex64
elif DTYPE == np.float64:
    COMPLEX_DTYPE = np.complex128
else:
    raise RuntimeError('')