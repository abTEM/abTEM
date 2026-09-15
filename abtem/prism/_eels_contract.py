"""Private, backend-preserving contractions for PRISM EELS reconstruction."""

from abtem.core.backend import get_array_module

# Internal override for correctness checks and benchmarks; not a public setting.
_EELS_CONTRACT_ORDER = "auto"


def _choose_contract_order(n_s2, n_k, n_active, n_pixels, n_transitions):
    """Prefer probe-first only when its arithmetic estimate is clearly cheaper."""
    # Count complex multiply-accumulates and pointwise products. Common output
    # scaling is omitted. The margin avoids switching for marginal estimates
    # that do not account for backend-dependent matrix multiplication overhead.
    beam_work = n_transitions * (
        n_k * n_pixels + n_s2 * n_pixels * n_k + n_s2 * n_k * n_active
    )
    probe_work = n_active * n_k * n_pixels + n_transitions * (
        n_active * n_pixels + n_s2 * n_pixels * n_active
    )
    return "probe" if 5 * probe_work < 4 * beam_work else "beam"


def contract_eels(s2_crop, h_crop, s1_crop, coefficients, scale, order="auto"):
    """Contract cropped EELS arrays into (transition, S2 beam, active probe).

    Inputs use shapes ``(n_s2, wy, wx)``, ``(n_T, wy, wx)``,
    ``(n_k, wy, wx)``, and ``(n_active, n_k)``. All arrays must share a
    NumPy or CuPy backend. Their configured complex precision is retained.
    ``beam`` preserves the original association; ``probe`` builds active
    probes first. Neither path builds a transition-by-beam spatial temporary.
    """
    if order not in ("auto", "beam", "probe"):
        raise ValueError("order must be 'auto', 'beam', or 'probe'")

    xp = get_array_module(s2_crop)
    n_s2 = s2_crop.shape[0]
    n_k = s1_crop.shape[0]
    n_transitions = h_crop.shape[0]
    n_active = coefficients.shape[0]
    n_pixels = s1_crop.shape[-2] * s1_crop.shape[-1]
    dtype = xp.result_type(
        s2_crop.dtype, h_crop.dtype, s1_crop.dtype, coefficients.dtype
    )
    result = xp.empty((n_transitions, n_s2, n_active), dtype=dtype)
    if n_active == 0 or n_transitions == 0:
        return result

    if order == "auto":
        order = _choose_contract_order(n_s2, n_k, n_active, n_pixels, n_transitions)

    s2_flat = s2_crop.conj().reshape(n_s2, n_pixels)
    s1_flat = s1_crop.reshape(n_k, n_pixels)
    h_flat = h_crop.reshape(n_transitions, n_pixels)
    if order == "probe":
        probes = coefficients @ s1_flat
        for transition in range(n_transitions):
            result[transition] = scale * (s2_flat @ (h_flat[transition] * probes).T)
    else:
        for transition in range(n_transitions):
            beam_amplitudes = scale * (s2_flat @ (h_flat[transition] * s1_flat).T)
            result[transition] = beam_amplitudes @ coefficients.T
    return result
