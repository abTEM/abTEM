"""Tests that abTEM reads its configuration from its own location.

The configuration is read once at import time, so each scenario runs in a fresh
interpreter with a controlled home directory and environment.
"""

import os
import subprocess
import sys

_SCRIPT = "import abtem; print(abtem.config.get('precision'))"


def _read_precision(tmp_path, **env):
    full_env = {
        k: v for k, v in os.environ.items() if not k.startswith(("ABTEM_", "DASK_"))
    }
    full_env["HOME"] = full_env["USERPROFILE"] = str(tmp_path / "home")
    full_env.update(env)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        env=full_env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def test_default_precision_is_float32(tmp_path):
    assert _read_precision(tmp_path) == "float32"


def test_yaml_in_user_config_directory_is_read(tmp_path):
    directory = tmp_path / "home" / ".config" / "abtem"
    directory.mkdir(parents=True)
    (directory / "abtem.yaml").write_text("precision: float64\n")

    assert _read_precision(tmp_path) == "float64"


def test_yaml_in_abtem_config_directory_is_read(tmp_path):
    directory = tmp_path / "custom"
    directory.mkdir()
    (directory / "abtem.yaml").write_text("precision: float64\n")

    assert _read_precision(tmp_path, ABTEM_CONFIG=str(directory)) == "float64"


def test_abtem_config_may_name_a_single_file(tmp_path):
    path = tmp_path / "my-settings.yaml"
    path.write_text("precision: float64\n")

    assert _read_precision(tmp_path, ABTEM_CONFIG=str(path)) == "float64"


def test_dask_environment_variables_are_still_read(tmp_path):
    assert _read_precision(tmp_path, DASK_PRECISION="float64") == "float64"


def test_environment_variables_take_precedence_over_yaml(tmp_path):
    directory = tmp_path / "custom"
    directory.mkdir()
    (directory / "abtem.yaml").write_text("precision: float64\n")

    precision = _read_precision(
        tmp_path, ABTEM_CONFIG=str(directory), DASK_PRECISION="float32"
    )
    assert precision == "float32"
