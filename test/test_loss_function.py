import numpy as np
import pytest

import abtem
from abtem.core.energy import energy2wavelength, relativistic_mass_correction
from abtem.inelastic.plasmons import characteristic_angle, drude_loss_function
from abtem.measurements import MeasurementsEnsemble

ENERGY = 200e3
BOHR = 0.529177210903


def m0v2(energy):
    gamma = relativistic_mass_correction(energy)
    return 510998.95 * (1 - 1 / gamma**2)


def drude(width=0.3, excitation_energy=16.7):
    e = np.linspace(0.01, 400.0, 40000)
    return e, drude_loss_function(e, excitation_energy, width)


def dispersive(q, e, excitation_energy=16.7, dispersion=200.0, width=1.0):
    """Drude peaks at E_p + dispersion * q^2, q in 1/Å."""
    return np.stack(
        [
            drude_loss_function(e, excitation_energy + dispersion * qi**2, width)
            for qi in q
        ]
    )


def write_gpaw_csv(path, omega, eps0=None, eps=None, eels0=None, eels=None):
    """Write a file in the format of gpaw.response.df.write_response_function."""
    with open(path, "w") as fd:
        if eels is not None:
            for w, a, b in zip(omega, eels0, eels):
                print(f"{w:.6f}, {a:.6f}, {b:.6f}", file=fd)
        else:
            for w, a, b in zip(omega, eps0, eps):
                print(
                    f"{w:.6f}, {a.real:.6f}, {a.imag:.6f}, {b.real:.6f}, {b.imag:.6f}",
                    file=fd,
                )


def test_angle_integral_without_momentum_dependence():
    e, loss = drude()
    loss_function = abtem.LossFunction(e, [0.0], loss)
    energies = np.array([5.0, 16.7, 30.0])
    theta_e = characteristic_angle(energies, ENERGY) * 1e-3
    expected = (
        drude_loss_function(energies, 16.7, 0.3)
        * np.log1p(0.02**2 / theta_e**2)
        / (np.pi * BOHR * m0v2(ENERGY))
    )
    result = loss_function.angle_integrated(energies, ENERGY, max_angle=20.0)
    assert np.allclose(result, expected, rtol=1e-4)

    with pytest.raises(ValueError, match="max_angle"):
        loss_function.mean_free_path(ENERGY)


def test_narrow_plasmon_reproduces_the_egerton_mean_free_path():
    e, loss = drude(width=0.05)
    loss_function = abtem.LossFunction(e, [0.0], loss)
    theta_e = characteristic_angle(16.7, ENERGY) * 1e-3
    gamma = relativistic_mass_correction(ENERGY)
    egerton = 2 * BOHR / (gamma * theta_e * np.log1p((0.0191 / theta_e) ** 2))
    assert np.isclose(
        loss_function.mean_free_path(ENERGY, max_angle=19.1), egerton, rtol=2e-3
    )
    assert np.isclose(loss_function.excitation_energy, 16.7, rtol=1e-3)


def test_critical_angle_of_a_loss_function_cut_at_that_angle():
    # constant in q up to the momentum transfer of 19.1 mrad, zero beyond: the
    # Lorentzian model with the same mean free path must be cut at 19.1 mrad
    e, loss = drude()
    q_cut = 19.1e-3 / energy2wavelength(ENERGY)
    loss_function = abtem.LossFunction(e, [0.0, q_cut], np.stack([loss, loss]))
    with pytest.warns(UserWarning, match="largest momentum transfer"):
        excitation_energy, critical_angle, mean_free_path = (
            loss_function.plasmon_parameters(ENERGY)
        )
    assert np.isclose(critical_angle, 19.1, rtol=1e-3)

    energies = np.array([10.0, 16.7, 25.0])
    q_e = characteristic_angle(energies, ENERGY) * 1e-3 / energy2wavelength(ENERGY)
    expected = (
        2
        / (np.pi * BOHR * m0v2(ENERGY))
        * drude_loss_function(energies, 16.7, 0.3)
        * np.log(q_cut / q_e)
    )
    assert np.allclose(
        loss_function.angle_integrated(energies, ENERGY), expected, rtol=1e-4
    )
    assert np.isclose(
        mean_free_path,
        loss_function.mean_free_path(ENERGY, max_angle=19.1),
        rtol=1e-3,
    )


def test_given_parameters_are_kept_and_the_critical_angle_refitted():
    q = np.linspace(0, 1.5, 61)
    e = np.linspace(0.05, 400, 4000)
    loss_function = abtem.LossFunction(
        e, q, dispersive(q, e) * np.exp(-q / 0.2)[:, None]
    )
    _, theta_c, mean_free_path = loss_function.plasmon_parameters(ENERGY)

    _, shorter_theta_c, given = loss_function.plasmon_parameters(
        ENERGY, mean_free_path=0.8 * mean_free_path
    )
    assert given == 0.8 * mean_free_path
    assert shorter_theta_c > theta_c

    assert loss_function.plasmon_parameters(ENERGY, critical_angle=5.0)[1] == 5.0
    assert loss_function.plasmon_parameters(ENERGY, excitation_energy=20.0)[0] == 20.0


@pytest.mark.parametrize(
    "model",
    [abtem.QuadraturePlasmons, abtem.MonteCarloPlasmons, abtem.PhaseScramblePlasmons],
)
def test_models_from_loss_function(model):
    q = np.linspace(0, 1.5, 61)
    e = np.linspace(0.05, 400, 4000)
    loss_function = abtem.LossFunction(
        e, q, dispersive(q, e) * np.exp(-q / 0.2)[:, None]
    )
    kwargs = {"num_excitations": 2} if model is abtem.MonteCarloPlasmons else {}
    plasmons = model.from_loss_function(loss_function, ENERGY, **kwargs)
    expected = loss_function.plasmon_parameters(ENERGY)
    parameters = plasmons.parameters
    assert np.allclose(
        [
            parameters["excitation_energy"],
            parameters["critical_angle"],
            parameters["mean_free_path"],
        ],
        expected,
    )

    plasmons = model.from_loss_function(
        loss_function, ENERGY, mean_free_path=1000.0, **kwargs
    )
    assert plasmons.parameters["mean_free_path"] == 1000.0
    assert np.isclose(
        plasmons.parameters["critical_angle"],
        loss_function.critical_angle(ENERGY, mean_free_path=1000.0),
    )


def test_spectrum_follows_the_plasmon_dispersion():
    q = np.linspace(0, 1.5, 61)
    e = np.linspace(0.05, 400, 8000)
    model = abtem.QuadraturePlasmons(
        mean_free_path=1000.0,
        excitation_energy=16.7,
        critical_angle=20.0,
        max_loss_order=1,
        num_angles=2,
        num_azimuthal=3,
        num_depths=2,
    )
    measurement = MeasurementsEnsemble(
        np.array([0.0, 1.0]), [model.order_axis], {"energy": ENERGY}
    )
    energies = np.arange(0.0, 60.0, 0.05)

    def peak(loss_function):
        spectrum = abtem.plasmon_loss_spectrum(
            measurement,
            energies,
            thickness=100.0,
            loss_function=loss_function,
            energy_resolution=0.0,
        )
        return energies[np.argmax(spectrum.array)]

    def mean_loss(loss_function):
        spectrum = abtem.plasmon_loss_spectrum(
            measurement,
            energies,
            thickness=100.0,
            loss_function=loss_function,
            energy_resolution=0.0,
        )
        return np.sum(energies * spectrum.array) / np.sum(spectrum.array)

    # small momentum transfers dominate (dq / q), so the dispersion mostly adds a
    # tail on the high-energy side of the peak
    flat = abtem.LossFunction(e, q, dispersive(q, e, dispersion=0.0))
    steep = abtem.LossFunction(e, q, dispersive(q, e, dispersion=200.0))
    assert abs(peak(flat) - 16.7) < 0.1
    assert abs(peak(steep) - 16.7) < 0.5
    assert mean_loss(steep) > mean_loss(flat) + 2.0

    # without momentum dependence it is the callable path
    single = abtem.LossFunction(e, [0.0], dispersive([0.0], e))
    a = abtem.plasmon_loss_spectrum(measurement, energies, 100.0, loss_function=single)
    b = abtem.plasmon_loss_spectrum(
        measurement,
        energies,
        100.0,
        loss_function=lambda x: drude_loss_function(x, 16.7, 1.0),
    )
    assert np.allclose(a.array, b.array, rtol=1e-3, atol=1e-6 * b.array.max())


@pytest.mark.parametrize("kind", ["eels", "df"])
def test_from_gpaw_files(tmp_path, kind):
    omega = np.linspace(0, 60, 601)
    q = np.array([0.0, 0.1, 0.2])
    w = omega[1:]
    # GPAW writes the static limit at omega = 0
    eps_lfc = np.r_[12.0, 1 - 16.7**2 / (w * (w + 1j))]
    eps_nlfc = np.r_[12.0, 1 - 17.5**2 / (w * (w + 1j))]
    files = []
    for i, qi in enumerate(q):
        path = tmp_path / f"{kind}_{i}.csv"
        scale = np.exp(-qi / 0.1)
        if kind == "eels":
            write_gpaw_csv(
                path,
                omega,
                eels0=scale * (-1 / eps_nlfc).imag,
                eels=scale * (-1 / eps_lfc).imag,
            )
        else:
            write_gpaw_csv(path, omega, eps0=eps_nlfc, eps=eps_lfc)
        files.append(str(path))

    # files out of order; q from the reduced q-points and the cell
    a = 5.0
    cell = np.diag([a, a, a])
    order = [2, 0, 1]
    loss_function = abtem.LossFunction.from_gpaw(
        [files[i] for i in order],
        q_points=[[q[i] * a, 0, 0] for i in order],
        cell=cell,
    )
    assert np.allclose(loss_function.momentum_transfers, q)
    expected = (-1 / eps_lfc).imag
    if kind == "eels":
        expected = np.exp(-q / 0.1)[:, None] * expected[None]
    assert np.allclose(loss_function.values, expected, atol=2e-6)
    assert np.isclose(loss_function.peak_energy, 16.7, atol=0.2)

    without = abtem.LossFunction.from_gpaw(
        files, momentum_transfers=q, local_field_effects=False
    )
    assert np.isclose(without.peak_energy, 17.5, atol=0.2)

    with pytest.warns(UserWarning, match="optical limit"):
        abtem.LossFunction.from_gpaw(files[1:], momentum_transfers=q[1:])


def test_from_gpaw_errors(tmp_path):
    path = tmp_path / "eels.csv"
    write_gpaw_csv(path, [0.0, 1.0], eels0=[0.0, 1.0], eels=[0.0, 1.0])
    other = tmp_path / "other.csv"
    write_gpaw_csv(other, [0.0, 2.0], eels0=[0.0, 1.0], eels=[0.0, 1.0])

    with pytest.raises(ValueError, match="one of"):
        abtem.LossFunction.from_gpaw([path])
    with pytest.raises(ValueError, match="cell"):
        abtem.LossFunction.from_gpaw([path], q_points=[[0, 0, 0]])
    with pytest.raises(ValueError, match="frequency grid"):
        abtem.LossFunction.from_gpaw([path, other], momentum_transfers=[0.0, 0.1])
    nan = tmp_path / "nan.csv"
    nan.write_text("0.000000, 0.000000, nan\n1.000000, 0.100000, nan\n")
    with pytest.raises(ValueError, match="reciprocal lattice vector"):
        abtem.LossFunction.from_gpaw([nan], momentum_transfers=[0.3])
    np.savetxt(tmp_path / "bad.csv", np.ones((3, 4)), delimiter=",")
    with pytest.raises(ValueError, match="columns"):
        abtem.LossFunction.from_gpaw([tmp_path / "bad.csv"], momentum_transfers=[0.0])


def test_write_read_round_trip(tmp_path):
    q = np.array([0.0, 0.5])
    e = np.linspace(0.1, 50, 100)
    loss_function = abtem.LossFunction(
        e, q, dispersive(q, e), metadata={"ecut": 50, "files": ["a.csv"]}
    )
    loss_function.write(tmp_path / "loss.npz")
    read = abtem.LossFunction.read(tmp_path / "loss.npz")
    assert np.array_equal(read.values, loss_function.values)
    assert np.array_equal(read.momentum_transfers, q)
    assert read.metadata == loss_function.metadata


def test_validation():
    e = np.linspace(0.1, 50, 10)
    with pytest.raises(ValueError, match="shape"):
        abtem.LossFunction(e, [0.0, 0.1], np.ones((3, 10)))
    with pytest.raises(ValueError, match="non-negative"):
        abtem.LossFunction(e, [0.0], -np.ones(10))
    with pytest.raises(ValueError, match="increasing"):
        abtem.LossFunction(e, [0.1, 0.0], np.ones((2, 10)))
    # small negative values from the broadening are clipped
    values = np.ones(10)
    values[0] = -1e-5
    assert abtem.LossFunction(e, [0.0], values).values.min() == 0.0


def write_gpaw_log(path, q_points, ecut):
    """A response log with the lines of GPAW's chi0 calculation that are read."""
    with open(path, "w") as fd:
        print("Using integrator: PointIntegrator", file=fd)
        for q_c in q_points:
            print("Calculating chi0 body with:", file=fd)
            print("    Ground state adapter containing:", file=fd)
            print("    Linear response parametrization:", file=fd)
            print(f"    q_c: {list(map(float, q_c))}", file=fd)
            print("    Number of frequency points: 3", file=fd)
            print(f"    Planewave cutoff: {ecut}", file=fd)
            print("    Number of bands: 90", file=fd)
            print("Finished calculating chi0", file=fd)


def test_from_gpaw_log_and_response_cutoff(tmp_path):
    omega = np.array([0.0, 10.0, 20.0])
    files = []
    for i in range(4):
        path = tmp_path / f"eels_{i}.csv"
        write_gpaw_csv(path, omega, eels0=[0, 1, 0], eels=[0, 1.0 / (i + 1), 0])
        files.append(path)
    a = 5.0
    cell = np.diag([a, a, a])
    q_points = [[0.0, 0, 0], [0.75, 0, 0], [0.25, 0, 0], [0.5, 0, 0]]
    log = tmp_path / "df.txt"
    write_gpaw_log(log, q_points, ecut=50.0)

    loss_function = abtem.LossFunction.from_gpaw(files, log=log, cell=cell)
    assert np.allclose(loss_function.momentum_transfers, [0, 0.05, 0.1, 0.15])
    assert np.allclose(loss_function.values[:, 1], [1, 1 / 3, 1 / 4, 1 / 2])
    assert loss_function.metadata["ecut"] == [50.0] * 4

    # |q| = 0.15 1/Å needs a cutoff above hbar^2 (2 pi q)^2 / 2m = 3.38 eV
    write_gpaw_log(log, q_points, ecut=3.3)
    with pytest.raises(ValueError, match="eels_1.csv.*beyond the response cutoff"):
        abtem.LossFunction.from_gpaw(files, log=log, cell=cell)
    abtem.LossFunction.from_gpaw(files, q_points=q_points, cell=cell, ecut=3.5)
    with pytest.raises(ValueError, match="beyond the response cutoff"):
        abtem.LossFunction.from_gpaw(files, q_points=q_points, cell=cell, ecut=3.3)

    with pytest.raises(ValueError, match="4 response calculations for 3 files"):
        abtem.LossFunction.from_gpaw(files[:3], log=log, cell=cell)
    with pytest.raises(ValueError, match="cell"):
        abtem.LossFunction.from_gpaw(files, log=log)


@pytest.mark.parametrize(
    "name, format", [("POSCAR", "vasp"), ("si.cif", "cif"), ("si.xyz", "extxyz")]
)
def test_from_gpaw_cell_from_structure_files(tmp_path, name, format):
    import ase.build
    import ase.io

    atoms = ase.build.bulk("Si", "diamond", a=5.431)
    ase.io.write(tmp_path / name, atoms, format=format)
    path = tmp_path / "eels.csv"
    write_gpaw_csv(path, [0.0, 10.0, 20.0], eels0=[0, 1, 0], eels=[0, 1, 0])

    loss_function = abtem.LossFunction.from_gpaw(
        [path, path], q_points=[[0, 0, 0], [0.25, 0, 0]], cell=tmp_path / name
    )
    b1 = np.linalg.norm(np.linalg.inv(atoms.cell).T[0])
    assert np.isclose(loss_function.momentum_transfers[1], 0.25 * b1, rtol=1e-5)


def test_from_gpaw_cell_from_file_without_lattice(tmp_path):
    import ase.build
    import ase.io

    ase.io.write(tmp_path / "plain.xyz", ase.build.molecule("H2O"), format="xyz")
    path = tmp_path / "eels.csv"
    write_gpaw_csv(path, [0.0, 10.0], eels0=[0, 1], eels=[0, 1])
    with pytest.raises(ValueError, match="plain.xyz has no unit cell"):
        abtem.LossFunction.from_gpaw(
            [path], q_points=[[0, 0, 0]], cell=str(tmp_path / "plain.xyz")
        )
