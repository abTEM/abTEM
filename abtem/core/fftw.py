import numpy as np
import pyfftw

FFTW_THREADS = 1
FFTW_EFFORT = 'FFTW_MEASURE'
FFTW_TIMELIMIT = 600


def create_fftw_objects(array, allow_new_plan=True):
    """
    Creates FFTW object for forward and backward Fourier transforms. The input array will be
    transformed in place. The function tries to retrieve FFTW plans from wisdom only.
    If no plan exists for the input array, a new plan is cached and then retrieved.

    """

    try:
        fftw_forward = pyfftw.FFTW(array, array, axes=(-1, -2),
                                   threads=FFTW_THREADS,
                                   flags=(FFTW_EFFORT, 'FFTW_WISDOM_ONLY', 'FFTW_DESTROY_INPUT'))
        fftw_backward = pyfftw.FFTW(array, array, axes=(-1, -2),
                                    direction='FFTW_BACKWARD', threads=FFTW_THREADS,
                                    flags=(FFTW_EFFORT, 'FFTW_WISDOM_ONLY', 'FFTW_DESTROY_INPUT'))
        return fftw_forward, fftw_backward

    except RuntimeError as e:
        if not allow_new_plan:
            fftw_forward = pyfftw.builders.fft2(array)
            fftw_backward = pyfftw.builders.ifft2(array)
            return fftw_forward, fftw_backward

        dummy = pyfftw.byte_align(np.zeros_like(array))

        pyfftw.FFTW(dummy, dummy,
                    axes=(-1, -2),
                    threads=FFTW_THREADS,
                    flags=(FFTW_EFFORT, 'FFTW_DESTROY_INPUT'),
                    planning_timelimit=FFTW_TIMELIMIT)

        pyfftw.FFTW(dummy, dummy, axes=(-1, -2),
                    direction='FFTW_BACKWARD',
                    threads=FFTW_THREADS,
                    flags=(FFTW_EFFORT, 'FFTW_DESTROY_INPUT'),
                    planning_timelimit=FFTW_TIMELIMIT)

        return create_fftw_objects(array, False)
