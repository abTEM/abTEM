import base64
import copy
import io
import os
import pprint
import re
from contextlib import contextmanager

import imageio.v3 as imageio
import nbformat
import numpy as np
from deepdiff import DeepDiff
from nbclient.client import NotebookClient


DEFAULT_NB_VERSION = 4

exclude_regex_paths = {
    r"root\['metadata'\]",
    r"root\['execution_count'\]",
    r"root\['outputs'\]\[\d+\]\['execution_count'\]",
}


# ROOT_DIR = "/Users/jacobmadsen/PycharmProjects/abtem-doc/docs/user_guide/"
ROOT_DIR = "C:\\Users\\jacob\\PycharmProjects\\abtem-docs\\docs\\user_guide"
KERNEL_NAME = "abtem-dask"


@contextmanager
def set_working_directory(directory):
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
                part[: match.span()[0] + 3] + "mem_addr" + part[match.span()[1] - 1 :]
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

    return nb


def bytes_to_image(b):
    decoded = base64.decodebytes(bytes(b, "utf-8"))
    bytes_io = io.BytesIO(decoded)
    return imageio.imread(bytes_io)


def _test_notebook(fname: str, working_directory: str, image_tolerance: float = 0.01):
    with set_working_directory(working_directory):
        nb = nbformat.read(fname, DEFAULT_NB_VERSION)
        nb_old = copy.deepcopy(nb)
        client = NotebookClient(nb, kernel=KERNEL_NAME)
        nb_new = client.execute()

    nb_old = strip_skipped(nb_old)
    nb_new = strip_skipped(nb_new)

    for i, (cell_old, cell_new) in enumerate(zip(nb_old["cells"], nb_new["cells"])):

        if "outputs" in cell_old.keys():
            assert "outputs" in cell_new
            for output_old, output_new in zip(cell_old["outputs"], cell_new["outputs"]):
                if "data" in output_old:
                    for key in output_old["data"].keys():
                        value_old = output_old["data"][key]
                        value_new = output_new["data"][key]

                        if key == "image/png":
                            image_old = bytes_to_image(value_old)
                            image_new = bytes_to_image(value_new)

                            errors = np.any(
                                np.abs(image_old - image_new), axis=-1
                            ).sum()
                            diff = errors / np.prod(image_old.shape[:-1])
                            if diff > image_tolerance:
                                diff = f"images not identical within tolerance {diff} > {image_tolerance}"
                                raise AssertionError(
                                    f"notebook changed in cell {i}:\n{diff}"
                                )
                else:
                    diff = DeepDiff(
                        output_old,
                        output_new,
                    )

                    if len(diff):
                        raise AssertionError(
                            f"notebook changed in cell - diff:\n{pprint.pformat(dict(diff))}"
                        )
        else:
            diff = DeepDiff(cell_old, cell_new, exclude_regex_paths=exclude_regex_paths)
            if len(diff):
                raise AssertionError(
                    f"notebook changed in cell - diff:\n{pprint.pformat(dict(diff))}"
                )
