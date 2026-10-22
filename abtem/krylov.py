"""Module for the Krylov multislice algorithm.

A multislice step advances the wave through one slice of thickness dz by

    psi -> exp(i dz F) psi,   F = sqrt(k^2 + laplacian + 2 k sigma V) - k,

where k = 2 pi / wavelength and V is the potential of the slice: the one-way (forward)
square root of the Helmholtz operator. The algorithms differ in how they evaluate it:

====================  ============================  =============================
                      potential outside the root    potential inside the root
====================  ============================  =============================
root exact            ``FourierMultislice``         ``KrylovMultislice``
                      (``order="exact"``)
root as a series      ``RealSpaceMultislice``       ``RealSpaceMultislice``
                      (``"propagator"``)            (``"full"``)
====================  ============================  =============================

The root of the free-space operator is diagonal in Fourier space, so it is a
multiplication. With the potential inside, the operator is diagonal in neither space,
and its exact root can only be applied to a wave as a function of the operator, here by
a Lanczos iteration in the Krylov space of the wave and the operator (one FFT pair per
iteration). The wave is kept in Fourier space and restricted to the propagating band,
where the operator is bounded and the root analytic; evanescent beams are dropped. The
truncated series of ``RealSpaceMultislice`` diverges when the grid reaches beyond the
band, which is the case at low energies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar

import numpy as np

from abtem.core import config
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.core.fft import fft2, ifft2
from abtem.finite_difference import NotConvergedError
from abtem.potentials.iam import TransmissionFunction
from abtem.tilt import _get_tilt_axes

if TYPE_CHECKING:
    from abtem.waves import Waves


@dataclass(frozen=True)
class KrylovMultislice:
    """
    Multislice algorithm with the exact one-way square root of the full operator.

    The potential is inside the square root and the root is evaluated exactly, by a
    Lanczos iteration in Fourier space (see the module documentation). The wave is
    restricted to the propagating band, ``|k| < propagating_cutoff / wavelength``, and
    each slice is advanced on the smallest grid that holds that band and the potential
    components that couple it. The result stays unitary and converges in the slice
    thickness; it is the algorithm to use between about 0.5 and 5 keV, where the split
    step is off by 10 % on the strong beams of graphene and the series of
    ``RealSpaceMultislice(expansion_scope="full")`` diverges. It is forward only: it has
    no reflection and no evanescent beams, which dominate the scattering below a few
    hundred eV.

    The potential must be given as projected potential slices, not as precomputed
    transmission functions, and should use finite projection with slices of about 0.05 Å
    or thinner. Tilted wave functions are not supported.

    Parameters
    ----------
    propagating_cutoff : float, optional
        Fraction of the wave number 1 / wavelength kept (default 0.95). Beams beyond it,
        evanescent and grazing, are removed from the wave.
        On grids too coarse for the band (high energies, where the band reaches far
        beyond the grid) the band is limited by the antialiasing aperture instead, as in
        the other algorithms.
    tolerance : float, optional
        Relative change of the Lanczos update at which a slice is converged (default
        1e-8).
    max_iterations : int, optional
        Maximum number of Lanczos iterations per slice (default 60).
    """

    propagating_cutoff: float = 0.95
    tolerance: float = 1e-8
    max_iterations: int = 60

    # Specified for compatibility with the code that selects the step of the two shipped
    # algorithms: it reads the expansion scope and never looks at transmission functions
    # (the transmission-function cache is built only for FourierMultislice).
    expansion_scope: ClassVar[str] = "propagator"

    def __post_init__(self):
        if not 0.0 < self.propagating_cutoff < 1.0:
            raise ValueError("propagating_cutoff must be between 0 and 1")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.max_iterations < 2:
            raise ValueError("max_iterations must be at least 2")

    def make_step(self, waves: Waves, return_backscattered: bool = False) -> Callable:
        """The per-slice step: ``exp(i dz F(O))`` applied by a Lanczos iteration."""
        if waves.base_tilt != (0.0, 0.0) or _get_tilt_axes(waves):
            raise NotImplementedError(
                "KrylovMultislice does not support tilted wave functions yet."
            )

        xp = get_array_module(waves.device)
        band = _Band(
            waves._valid_gpts,
            waves._valid_sampling,
            waves._valid_energy,
            self.propagating_cutoff,
            xp,
        )

        def step(waves, potential_slice, next_slice=None):
            if isinstance(potential_slice, TransmissionFunction):
                raise TypeError(
                    "KrylovMultislice needs the projected potential of each slice, "
                    "not precomputed transmission functions."
                )
            if waves.device != potential_slice.device:
                potential_slice = potential_slice.copy_to_device(device=waves.device)

            thickness = potential_slice.thickness
            sigma_v = (
                potential_slice.array[0] * energy2sigma(waves._valid_energy) / thickness
            )
            waves._array = band.advance(
                waves._array,
                sigma_v,
                thickness,
                self.tolerance,
                self.max_iterations,
            )
            return waves

        return step


class _Band:
    """The propagating band of one grid and energy, and the grid that holds it."""

    def __init__(self, gpts, sampling, energy, cutoff, xp):
        self.xp = xp
        wavelength = energy2wavelength(energy)
        self.k = 2 * np.pi / wavelength
        nx, ny = gpts
        lx, ly = nx * sampling[0], ny * sampling[1]
        kc = cutoff / wavelength

        # On a grid too coarse for the propagating band (high energies) the band is
        # limited as in the other algorithms, by the antialiasing aperture. A product
        # of the wave and the potential then lands outside the band when the potential
        # is limited to 2 nyquist - 2 kc, which is kc itself at the aperture.
        nyquist = 1 / (2 * max(sampling))
        kc = min(kc, config.get("antialias.cutoff") * nyquist)
        kv = min(2 * kc, 2 * nyquist - 2 * kc)

        # The band |k| < kc and the potential components that couple it, |q| < kv, fit
        # a grid whose Nyquist frequency is kv. It is never larger than the grid given.
        mx = min(nx, 2 * int(np.ceil(kv * lx)))
        my = min(ny, 2 * int(np.ceil(kv * ly)))
        self.fine_gpts, self.coarse_gpts = (nx, ny), (mx, my)
        ix = np.fft.fftfreq(mx, 1 / mx).round().astype(int)
        iy = np.fft.fftfreq(my, 1 / my).round().astype(int)
        self.rows = xp.asarray((ix % nx)[:, None])
        self.cols = xp.asarray((iy % ny)[None, :])

        k2 = (ix[:, None] / lx) ** 2 + (iy[None, :] / ly) ** 2
        self.projector = xp.asarray(k2 < kc**2)
        self.potential_window = xp.asarray(k2 < kv**2)
        # laplacian / (4 pi K) = -pi k^2 / K in Fourier space, in the band
        self.kinetic = xp.asarray(-np.pi * k2 * wavelength * (k2 < kc**2))

    def advance(self, array, sigma_v, thickness, tolerance, max_iterations):
        """exp(i dz F(O)) applied to wave functions given as real-space arrays."""
        xp = self.xp
        mx, my = self.coarse_gpts
        nx, ny = self.fine_gpts
        real_dtype = np.empty(0, dtype=array.dtype).real.dtype

        wave_k = fft2(array)
        a0 = wave_k[..., self.rows, self.cols]

        # the slice potential on the coarse grid: filtered, so nothing aliases into the
        # band
        v_k = (
            fft2(sigma_v.astype(array.dtype))[self.rows, self.cols]
            * self.potential_window
        )
        potential = ifft2(v_k).real * (mx * my / (nx * ny))

        kinetic = self.kinetic.astype(real_dtype)
        projector = self.projector

        def apply(a):
            return kinetic * a + projector * fft2(potential * ifft2(a))

        batch_shape = a0.shape[:-2]
        a1 = _lanczos_exponential(
            a0.reshape((-1, mx, my)),
            apply,
            self.k,
            thickness,
            tolerance,
            max_iterations,
            xp,
        )

        out_k = xp.zeros_like(wave_k)
        out_k[..., self.rows, self.cols] = a1.reshape(batch_shape + (mx, my))
        return ifft2(out_k)


def _inner(a, b, xp):
    """Per-member inner product over the last two axes, shape (B,)."""
    return xp.sum(xp.conj(a) * b, axis=(-2, -1))


def _lanczos_exponential(a0, apply, k, dz, tolerance, max_iterations, xp):
    """exp(i dz f(O)) a0 for every member of a0, with f(t) = k (sqrt(1 + 2 t / k) - 1).

    One application of the Hermitian operator ``apply`` per iteration. The small
    tridiagonal matrices are diagonalised on the host, in double precision.
    """
    dtype = a0.dtype
    real_dtype = np.empty(0, dtype=dtype).real.dtype
    norm0 = xp.sqrt(_inner(a0, a0, xp).real)
    norm0_host = asnumpy(norm0).astype(np.float64)
    safe = xp.asarray(np.where(norm0_host > 0, norm0_host, 1.0).astype(real_dtype))
    q = a0 / safe[:, None, None]

    basis = [q]
    alphas, betas = [], []
    coefficients_previous = None

    for m in range(1, max_iterations + 1):
        w = apply(basis[-1])
        alpha = asnumpy(_inner(basis[-1], w, xp).real).astype(np.float64)
        w = w - xp.asarray(alpha.astype(real_dtype))[:, None, None] * basis[-1]
        if betas:
            w = w - xp.asarray(betas[-1].astype(real_dtype))[:, None, None] * basis[-2]
        for vector in basis:  # full reorthogonalisation: m is small
            w = w - _inner(vector, w, xp)[:, None, None] * vector
        alphas.append(alpha)

        size = len(alphas)
        tridiagonal = np.zeros((a0.shape[0], size, size))
        index = np.arange(size)
        tridiagonal[:, index, index] = np.stack(alphas, axis=1)
        if betas:
            off = np.stack(betas, axis=1)
            tridiagonal[:, index[:-1], index[1:]] = off
            tridiagonal[:, index[1:], index[:-1]] = off
        values, vectors = np.linalg.eigh(tridiagonal)

        argument = 1.0 + 2.0 * values / k
        if argument.min() <= 0.0:
            raise NotImplementedError(
                "The operator has eigenvalues at or below the edge of the propagating "
                "band (an exceptionally strong potential); lower propagating_cutoff or "
                "use a "
                "finer slicing."
            )
        phase = np.exp(1j * dz * k * (np.sqrt(argument) - 1.0))
        coefficients = np.einsum(
            "bij,bj,bj->bi", vectors, phase, vectors[:, 0, :].conj()
        )

        if coefficients_previous is not None:
            padded = np.pad(coefficients_previous, ((0, 0), (0, 1)))
            if np.linalg.norm(coefficients - padded, axis=1).max() < tolerance:
                break

        beta = asnumpy(xp.sqrt(_inner(w, w, xp).real)).astype(np.float64)
        if (beta < 1e-14).all():
            break
        betas.append(beta)
        basis.append(
            w
            / xp.asarray(np.where(beta > 1e-14, beta, 1.0).astype(real_dtype))[
                :, None, None
            ]
        )
        coefficients_previous = coefficients
    else:
        raise NotConvergedError(
            f"the Lanczos iteration did not converge to {tolerance} in "
            f"{max_iterations} iterations"
        )

    result = xp.zeros_like(a0)
    for j, vector in enumerate(basis[: coefficients.shape[1]]):
        weight = xp.asarray((coefficients[:, j] * norm0_host).astype(dtype))
        result = result + weight[:, None, None] * vector
    return result
