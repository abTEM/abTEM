import copy
import os
import pprint
import re
from contextlib import contextmanager

import nbformat
from deepdiff import DeepDiff
from nbclient.client import NotebookClient

DEFAULT_NB_VERSION = 4

exclude_regex_paths = {
    r"root\['cells'\]\[\d+\]\['metadata'\]",
    r"root\['cells'\]\[\d+\]\['execution_count'\]",
    r"root\['cells'\]\[\d+\]\['outputs'\]\[\d+\]\['execution_count'\]",
}

# ROOT_DIR = '../../docs/user_guide/'
ROOT_DIR = "/Users/jacobmadsen/PycharmProjects/abtem-doc/docs/user_guide/walkthrough"


@contextmanager
def working_directory(directory):
    owd = os.getcwd()
    try:
        os.chdir(directory)
        yield directory
    finally:
        os.chdir(owd)


def apply_func_to_code_output(nb, func, output_type):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = [
                func(output) if output["output_type"] == output_type else output
                for output in cell["outputs"]
            ]
    return nb


def strip_memory_address(text):
    parts = re.split("<", text)
    for i, part in enumerate(parts):
        match = re.search("at .+>", part)
        if match:
            parts[i] = (
                    part[: match.span()[0] + 3] + "mem_addr" + part[match.span()[1] - 1:]
            )
    return "<".join(parts)


def strip_memory_address_from_code_output(nb):
    def apply_replace_memory_address_execute_result(output):
        output["data"]["text/plain"] = strip_memory_address(
            output["data"]["text/plain"]
        )
        return output

    def apply_replace_memory_address_stream(output):
        output["text"] = strip_memory_address(output["text"])
        return output

    nb = apply_func_to_code_output(
        nb, apply_replace_memory_address_execute_result, "execute_result"
    )
    nb = apply_func_to_code_output(nb, apply_replace_memory_address_stream, "stream")
    return nb


def strip_timings_from_code_output(nb):
    def apply_replace_memory_address_execute_result(output):
        output["data"]["text/plain"] = strip_memory_address(
            output["data"]["text/plain"]
        )
        return output

    def apply_replace_memory_address_stream(output):
        output["text"] = strip_memory_address(output["text"])
        return output

    nb = apply_func_to_code_output(
        nb, apply_replace_memory_address_execute_result, "execute_result"
    )
    nb = apply_func_to_code_output(nb, apply_replace_memory_address_stream, "stream")
    return nb


def strip_skipped(nb):
    for cell in nb["cells"]:
        if (cell["cell_type"] == "code") and (
                "skip-test" in cell["metadata"].get("tags", [])
        ):
            cell["outputs"] = []


def _test_notebook(fname):
    with working_directory(ROOT_DIR):
        nb = nbformat.read(fname, DEFAULT_NB_VERSION)
        nb_old = copy.deepcopy(nb)
        client = NotebookClient(nb)
        nb_new = client.execute()

    nb_old = strip_skipped(nb_old)
    nb_new = strip_skipped(nb_new)

    diff = DeepDiff(nb_old, nb_new, exclude_regex_paths=exclude_regex_paths)

    if len(diff) != 0:
        raise AssertionError(f"notebook changed - diff:\n{pprint.pformat(dict(diff))}")


def test_atomic_models():
    _test_notebook("atomic_models.ipynb")


def test_potentials():
    _test_notebook("potentials.ipynb")


def test_wave_functions():
    _test_notebook("wave_functions.ipynb")


def test_multislice():
    _test_notebook("multislice.ipynb")


def test_contrast_transfer_function():
    _test_notebook("contrast_transfer_function.ipynb")


def test_scan_and_detect():
    _test_notebook("scan_and_detect.ipynb")


def test_frozen_phonons():
    _test_notebook("frozen_phonons.ipynb")
