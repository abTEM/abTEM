"""Tests that abTEM reads configuration from its own location.

abTEM's config module is modelled on dask's, but for a long time ``refresh``
delegated to :func:`dask.config.collect` with no arguments, so abTEM only ever
saw dask's config paths and dask's ``DASK_``-prefixed environment variables --
the ``ABTEM_CONFIG``/``~/.config/abtem`` location was computed and then never
used.
"""

import os
import subprocess
import sys
import textwrap

import dask.config
import pytest

from abtem.core import config


def write_config(directory, text):
    path = directory / "abtem.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


@pytest.fixture
def fresh_config():
    """An isolated config dict seeded from abTEM's own defaults."""
    return {}


def refresh_into(target, paths=(), env=None):
    """Refresh ``target`` from ``paths`` only, ignoring the ambient environment."""
    config.refresh(config=target, paths=list(paths), env={} if env is None else env)


class TestPaths:
    def test_user_config_dir_is_searched(self):
        assert os.path.join(os.path.expanduser("~"), ".config", "abtem") in config.paths

    def test_abtem_config_env_var_appends_highest_priority_path(self, monkeypatch):
        monkeypatch.setenv("ABTEM_CONFIG", "/somewhere/abtem-config")
        paths = config._get_paths()
        assert paths[-1] == "/somewhere/abtem-config"

    def test_abtem_root_config_env_var_replaces_system_path(self, monkeypatch):
        monkeypatch.setenv("ABTEM_ROOT_CONFIG", "/somewhere/etc-abtem")
        paths = config._get_paths()
        assert paths[0] == "/somewhere/etc-abtem"
        assert "/etc/abtem" not in paths

    def test_dask_config_paths_are_not_searched(self):
        """abTEM no longer picks up dask's config files as its own."""
        assert not set(dask.config.paths) & set(config.paths)


class TestYamlFiles:
    def test_config_file_in_abtem_path_is_read(self, tmp_path, fresh_config):
        write_config(tmp_path, "device: gpu")

        refresh_into(fresh_config, paths=[tmp_path])

        assert config.get("device", config=fresh_config) == "gpu"

    def test_user_config_overrides_defaults_and_leaves_the_rest(
        self, tmp_path, fresh_config
    ):
        write_config(
            tmp_path,
            """
            device: gpu
            dask:
              chunk-size: 999 MB
            """,
        )

        refresh_into(fresh_config, paths=[tmp_path])

        assert config.get("device", config=fresh_config) == "gpu"
        assert config.get("dask.chunk-size", config=fresh_config) == "999 MB"
        # Untouched keys keep the shipped defaults ...
        assert config.get("precision", config=fresh_config) == "float32"
        # ... including siblings of an overridden nested key.
        assert config.get("dask.lazy", config=fresh_config) is True

    def test_later_paths_win(self, tmp_path, fresh_config):
        low, high = tmp_path / "low", tmp_path / "high"
        low.mkdir()
        high.mkdir()
        write_config(low, "device: gpu")
        write_config(high, "device: cpu")

        refresh_into(fresh_config, paths=[low, high])

        assert config.get("device", config=fresh_config) == "cpu"

    def test_missing_paths_are_ignored(self, tmp_path, fresh_config):
        refresh_into(fresh_config, paths=[tmp_path / "does-not-exist"])

        assert config.get("device", config=fresh_config) == "cpu"


class TestEnvironmentVariables:
    def test_abtem_prefix_is_read(self):
        collected = config.collect_env({"ABTEM_DEVICE": "gpu"})

        assert collected == {"device": "gpu"}

    def test_double_underscore_is_nested_access(self):
        collected = config.collect_env({"ABTEM_DASK__CHUNK_SIZE": "256 MB"})

        assert collected == {"dask": {"chunk_size": "256 MB"}}

    def test_values_are_literal_evaluated(self):
        collected = config.collect_env({"ABTEM_FFTW__THREADS": "8"})

        assert collected == {"fftw": {"threads": 8}}

    def test_unprefixed_variables_are_ignored(self):
        assert config.collect_env({"DEVICE": "gpu", "PATH": "/usr/bin"}) == {}

    def test_env_overrides_yaml(self, tmp_path, fresh_config):
        write_config(tmp_path, "device: gpu")

        refresh_into(fresh_config, paths=[tmp_path], env={"ABTEM_DEVICE": "cpu"})

        assert config.get("device", config=fresh_config) == "cpu"

    def test_underscore_env_key_reaches_hyphenated_default(self, fresh_config):
        refresh_into(fresh_config, env={"ABTEM_DASK__CHUNK_SIZE": "256 MB"})

        assert config.get("dask.chunk-size", config=fresh_config) == "256 MB"
        # and does not leave a second, underscored copy behind
        assert "chunk_size" not in fresh_config["dask"]


class TestControlEnvVarsDoNotLeak:
    """ABTEM_CONFIG/ABTEM_ROOT_CONFIG control config *discovery* (_get_paths);
    they must not also be read as configuration values by collect_env, or
    they'd land in the config dict as stray `config`/`root_config` keys."""

    def test_abtem_config_does_not_leak(self):
        assert config.collect_env({"ABTEM_CONFIG": "/some/dir"}) == {}

    def test_abtem_root_config_does_not_leak(self):
        assert config.collect_env({"ABTEM_ROOT_CONFIG": "/some/dir"}) == {}

    def test_real_keys_still_pass_through(self):
        # Sanity check that the exclusion is scoped to the two control
        # variables, not to ABTEM_ variables in general.
        assert config.collect_env({"ABTEM_DEVICE": "gpu"}) == {"device": "gpu"}


class TestMalformedYaml:
    def test_malformed_yaml_warns_and_falls_back_instead_of_crashing_import(
        self, tmp_path, fresh_config
    ):
        (tmp_path / "abtem.yaml").write_text(
            "device: [unterminated\n  - this is not valid yaml: :\n",
            encoding="utf-8",
        )

        with pytest.warns(RuntimeWarning, match="abtem.yaml|malformed"):
            refresh_into(fresh_config, paths=[tmp_path])

        # Falls back to the shipped default rather than raising.
        assert config.get("device", config=fresh_config) == "cpu"


class TestLegacyDaskEnvironmentVariables:
    def test_abtem_key_still_works_but_warns(self):
        with pytest.warns(FutureWarning, match="ABTEM_DEVICE"):
            collected = config.collect_legacy_env({"DASK_DEVICE": "gpu"})

        assert collected == {"device": "gpu"}

    def test_nested_abtem_key_still_works(self):
        with pytest.warns(FutureWarning, match="ABTEM_DASK__CHUNK_SIZE"):
            collected = config.collect_legacy_env({"DASK_DASK__CHUNK_SIZE": "256 MB"})

        assert collected == {"dask": {"chunk_size": "256 MB"}}

    @pytest.mark.parametrize(
        "name",
        [
            "DASK_DISTRIBUTED__WORKER__MEMORY__TARGET",
            "DASK_ARRAY__CHUNK_SIZE",
            "DASK_TEMPORARY_DIRECTORY",
            "DASK_CONFIG",
        ],
    )
    def test_genuine_dask_variables_are_left_to_dask(self, name, recwarn):
        """Keys abTEM does not define must neither warn nor enter its config."""
        assert config.collect_legacy_env({name: "0.8"}) == {}
        assert not [w for w in recwarn if w.category is FutureWarning]

    def test_abtem_prefix_wins_over_dask_prefix(self, fresh_config):
        with pytest.warns(FutureWarning):
            refresh_into(
                fresh_config,
                env={"DASK_DEVICE": "gpu", "ABTEM_DEVICE": "cpu"},
            )

        assert config.get("device", config=fresh_config) == "cpu"


class TestRuntimeAccess:
    """Setting values on the live config dict is unchanged by all of the above."""

    def test_set_as_context_manager(self):
        before = config.get("device")
        with config.set({"device": "gpu"}):
            assert config.get("device") == "gpu"
        assert config.get("device") == before

    def test_nested_set_with_kwargs(self):
        with config.set(fftw__threads=13):
            assert config.get("fftw.threads") == 13

    def test_defaults_are_loaded_at_import(self):
        assert config.get("device") in ("cpu", "gpu")
        assert config.get("fft") in ("numpy", "fftw", "mkl")


@pytest.mark.parametrize(
    "env_var, expected",
    [("ABTEM_CONFIG", "gpu"), ("DASK_CONFIG", "cpu")],
)
def test_config_file_is_read_at_import(tmp_path, env_var, expected):
    """End-to-end: the reported bug was about ``import abtem`` itself.

    ``ABTEM_CONFIG`` must be honored; ``DASK_CONFIG`` must no longer be, so a
    stray yaml file in dask's config directory cannot change abTEM's settings.
    """
    write_config(tmp_path, "device: gpu")

    env = {
        k: v for k, v in os.environ.items() if k not in ("ABTEM_CONFIG", "DASK_CONFIG")
    }
    env[env_var] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", "import abtem; print(abtem.config.get('device'))"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert result.stdout.strip() == expected
