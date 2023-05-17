import os

import pytest

from utils import ROOT_DIR, _test_notebook


@pytest.mark.slow
def test_atomic_models():
    _test_notebook("atomic_models.ipynb", os.path.join(ROOT_DIR, "walkthrough"))


@pytest.mark.slow
def test_potentials():
    _test_notebook("potentials.ipynb", os.path.join(ROOT_DIR, "walkthrough"))


@pytest.mark.slow
def test_wave_functions():
    _test_notebook("wave_functions.ipynb", os.path.join(ROOT_DIR, "walkthrough"))


@pytest.mark.slow
def test_multislice():
    _test_notebook("multislice.ipynb", os.path.join(ROOT_DIR, "walkthrough"))


@pytest.mark.slow
def test_contrast_transfer_function():
    _test_notebook(
        "contrast_transfer_function.ipynb", os.path.join(ROOT_DIR, "walkthrough")
    )


@pytest.mark.slow
def test_scan_and_detect():
    _test_notebook("scan_and_detect.ipynb", os.path.join(ROOT_DIR, "walkthrough"))


@pytest.mark.slow
def test_frozen_phonons():
    _test_notebook("frozen_phonons.ipynb", os.path.join(ROOT_DIR, "walkthrough"))


@pytest.mark.slow
def test_parallelization():
    _test_notebook("parallelization.ipynb", os.path.join(ROOT_DIR, "walkthrough"))


@pytest.mark.slow
def test_advanced_atomic_models():
    _test_notebook("advanced_atomic_models.ipynb", os.path.join(ROOT_DIR, "tutorials"))


@pytest.mark.slow
def test_partial_coherence():
    _test_notebook("partial_coherence.ipynb", os.path.join(ROOT_DIR, "tutorials"))


@pytest.mark.slow
def test_prism():
    _test_notebook("prism.ipynb", os.path.join(ROOT_DIR, "tutorials"))


@pytest.mark.slow
def test_epie():
    _test_notebook("epie.ipynb", os.path.join(ROOT_DIR, "tutorials"))


# @pytest.mark.slow
# def test_charge_density():
#     _test_notebook("charge_density.ipynb", os.path.join(ROOT_DIR, "tutorials"))
#
