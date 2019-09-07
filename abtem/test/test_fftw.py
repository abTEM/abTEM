# temp = pyfftw.empty_aligned(wave._array.shape, dtype='complex64')
# wave._array[:] = wave._array * complex_exponential(wave.sigma * potential_slice)
#
# fft_object_forward = pyfftw.FFTW(wave._array, temp, axes=(1, 2), threads=2)
# fft_object_backward = pyfftw.FFTW(temp, wave._array, axes=(1, 2), threads=2, direction='FFTW_BACKWARD')
#
# temp[:] = fft_object_forward() * propagator
# fft_object_backward()