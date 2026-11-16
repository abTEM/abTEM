"""``units`` of ``show()`` converts the plotted data, not only the axis label."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import abtem
from abtem.core.axes import ParameterAxis
from abtem.core.energy import energy2wavelength
from abtem.measurements import RealSpaceLineProfiles, ReciprocalSpaceLineProfiles

ENERGY = 100e3


def _drawn_line(visualization):
    ax = plt.gcf().axes[0]
    x = np.array(ax.get_lines()[0].get_xdata())
    label = ax.get_xlabel()
    plt.close("all")
    return x, label


def _reciprocal_space_profiles():
    return ReciprocalSpaceLineProfiles(
        np.arange(6.0), sampling=0.25, metadata={"energy": ENERGY}
    )


def test_reciprocal_space_profiles_are_shown_in_mrad():
    profiles = _reciprocal_space_profiles()

    k, k_label = _drawn_line(profiles.show())
    alpha, alpha_label = _drawn_line(profiles.show(units="mrad"))

    np.testing.assert_allclose(k, 0.25 * np.arange(6))
    np.testing.assert_allclose(alpha, k * energy2wavelength(ENERGY) * 1e3)
    assert "mrad" in alpha_label
    assert "mrad" not in k_label


def test_real_space_profiles_are_shown_in_nm():
    profiles = RealSpaceLineProfiles(np.arange(6.0), sampling=0.5)

    angstrom, _ = _drawn_line(profiles.show())
    nanometer, label = _drawn_line(profiles.show(units="nm"))

    np.testing.assert_allclose(nanometer, angstrom * 0.1)
    assert "nm" in label


@pytest.mark.parametrize("units", [None, "1/Å", "mrad"])
def test_ctf_profiles_follow_the_requested_units(units):
    ctf = abtem.CTF(energy=200e3, defocus=50, semiangle_cutoff=20)
    ctf.grid.extent = 20
    ctf.grid.sampling = 0.1
    k = ctf.profiles().show()
    k_max = np.max(_drawn_line(k)[0])

    x, _ = _drawn_line(ctf.profiles().show(units=units))

    expected = k_max * (energy2wavelength(200e3) * 1e3 if units == "mrad" else 1.0)
    assert np.max(x) == pytest.approx(expected)


def test_the_axis_of_an_ensemble_is_converted():
    ensemble = abtem.MeasurementsEnsemble(
        np.arange(3.0),
        ensemble_axes_metadata=[
            ParameterAxis(values=(0.0, 1.0, 2.0), label="alpha", units="mrad")
        ],
    )

    x, label = _drawn_line(ensemble.show(units="rad", overlay=True))

    np.testing.assert_allclose(x, [0.0, 1e-3, 2e-3])
    assert "rad" in label and "mrad" not in label


def test_incompatible_units_are_refused_rather_than_only_relabelled():
    profiles = RealSpaceLineProfiles(np.arange(6.0), sampling=0.5)

    with pytest.raises(RuntimeError, match="cannot convert units"):
        profiles.show(units="mrad")

    plt.close("all")


def test_angles_need_the_energy_to_be_known():
    profiles = ReciprocalSpaceLineProfiles(np.arange(6.0), sampling=0.25)

    with pytest.raises(RuntimeError, match="energy must be provided"):
        profiles.show(units="mrad")

    plt.close("all")


def test_updating_the_data_of_a_converted_profile_keeps_the_units():
    profiles = _reciprocal_space_profiles()
    visualization = profiles.show(units="mrad")
    artist = np.asarray(visualization.artists).ravel()[0]
    before = np.array(artist._lines[0].get_xdata())

    artist.set_data(profiles * 2)

    np.testing.assert_allclose(artist._lines[0].get_xdata(), before)
    plt.close("all")
