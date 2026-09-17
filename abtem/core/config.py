from __future__ import annotations

import os
import site
import sys
import threading
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Union

import yaml  # type: ignore
from dask.config import canonical_name, collect_yaml, interpret_value, merge, update

no_default = "__no_default__"

#: Prefix of the environment variables abTEM reads its configuration from,
#: e.g. ``ABTEM_DEVICE=gpu`` or ``ABTEM_DASK__CHUNK_SIZE="256 MB"``.
ENV_PREFIX = "ABTEM_"

#: abTEM used to read its configuration from dask's environment namespace by
#: accident. Variables in that namespace are still honored, but only for keys
#: abTEM actually defines, and they warn. See :func:`collect_legacy_env`.
LEGACY_ENV_PREFIX = "DASK_"


def _get_paths() -> list[str]:
    """Get locations to search for YAML configuration files.

    This logic exists as a separate function for testing purposes.
    """
    paths = [
        os.getenv("ABTEM_ROOT_CONFIG", "/etc/abtem"),
        os.path.join(sys.prefix, "etc", "abtem"),
        *[os.path.join(prefix, "etc", "abtem") for prefix in site.PREFIXES],
        os.path.join(os.path.expanduser("~"), ".config", "abtem"),
    ]
    if "ABTEM_CONFIG" in os.environ:
        paths.append(os.environ["ABTEM_CONFIG"])

    # Remove duplicate paths while preserving ordering
    paths = list(reversed(list(dict.fromkeys(reversed(paths)))))

    return paths


paths = _get_paths()

#: Environment variables that control config *discovery* itself (see
#: :func:`_get_paths`) rather than naming a configuration value. Excluded from
#: :func:`collect_env` so they don't leak into the config dict as stray
#: top-level keys (``config``, ``root_config``).
_CONTROL_ENV_VARS = frozenset({"ABTEM_CONFIG", "ABTEM_ROOT_CONFIG"})

config: dict = {}

config_lock = threading.Lock()

defaults: list[Mapping] = []


class set:
    """Temporarily set configuration values within a context manager

    Parameters
    ----------
    arg : mapping or None, optional
        A mapping of configuration key-value pairs to set.
    **kwargs :
        Additional key-value pairs to set. If ``arg`` is provided, values set
        in ``arg`` will be applied before those in ``kwargs``.
        Double-underscores (``__``) in keyword arguments will be replaced with
        ``.``, allowing nested values to be easily set.
    """

    config: dict
    # [(op, path, value), ...]
    _record: list[tuple[Literal["insert", "replace"], tuple[str, ...], Any]]

    def __init__(
        self,
        arg: Union[Mapping, None] = None,
        config: dict = config,
        lock: threading.Lock = config_lock,
        **kwargs,
    ):
        with lock:
            self.config = config
            self._record = []

            if arg is not None:
                for key, value in arg.items():
                    key = check_deprecations(key)
                    self._assign(key.split("."), value, config)
            if kwargs:
                for key, value in kwargs.items():
                    key = key.replace("__", ".")
                    key = check_deprecations(key)
                    self._assign(key.split("."), value, config)

    def __enter__(self):
        return self.config

    def __exit__(self, type, value, traceback):
        for op, path, value in reversed(self._record):
            d = self.config
            if op == "replace":
                for key in path[:-1]:
                    d = d.setdefault(key, {})
                d[path[-1]] = value
            else:  # insert
                for key in path[:-1]:
                    try:
                        d = d[key]
                    except KeyError:
                        break
                else:
                    d.pop(path[-1], None)

    def _assign(
        self,
        keys: Sequence[str],
        value: Any,
        d: dict,
        path: tuple[str, ...] = (),
        record: bool = True,
    ) -> None:
        """Assign value into a nested configuration dictionary

        Parameters
        ----------
        keys : Sequence[str]
            The nested path of keys to assign the value.
        value : object
        d : dict
            The part of the nested dictionary into which we want to assign the
            value
        path : tuple[str], optional
            The path history up to this point.
        record : bool, optional
            Whether this operation needs to be recorded to allow for rollback.
        """
        key = canonical_name(keys[0], d)

        path = path + (key,)

        if len(keys) == 1:
            if record:
                if key in d:
                    self._record.append(("replace", path, d[key]))
                else:
                    self._record.append(("insert", path, None))
            d[key] = value
        else:
            if key not in d:
                if record:
                    self._record.append(("insert", path, None))
                d[key] = {}
                # No need to record subsequent operations after an insert
                record = False
            self._assign(keys[1:], value, d[key], path, record=record)


def _defines_key(key: str, defaults: list[Mapping] = defaults) -> bool:
    """Whether ``key`` (in dotted form) is a key defined by abTEM's defaults."""
    for default in defaults:
        node: Any = default
        for part in key.split("."):
            # dict rather than Mapping: the defaults are parsed from yaml, and
            # canonical_name is typed for dict.
            if not isinstance(node, dict):
                break
            part = canonical_name(part, node)
            if part not in node:
                break
            node = node[part]
        else:
            return True
    return False


def collect_env(env: Mapping[str, str] | None = None) -> dict:
    """Collect config from environment variables

    This grabs environment variables of the form ``ABTEM_FOO__BAR_BAZ=123`` and
    turns these into config variables of the form ``{"foo": {"bar-baz": 123}}``.
    It transforms the key and value in the following way:

    -  Lower-cases the key text
    -  Treats ``__`` (double-underscore) as nested access
    -  Calls ``ast.literal_eval`` on the value

    ``ABTEM_CONFIG`` and ``ABTEM_ROOT_CONFIG`` are excluded even though they
    carry the ``ABTEM_`` prefix: they control where :func:`collect` looks for
    yaml files (see :func:`_get_paths`) rather than naming a configuration
    value, and must not leak into the config dict as a stray ``config`` or
    ``root_config`` key.

    Parameters
    ----------
    env : Mapping[str, str], optional
        The system environment variables. Defaults to ``os.environ``.

    Returns
    -------
    config : dict
    """
    if env is None:
        env = os.environ

    d = {
        name[len(ENV_PREFIX) :].lower().replace("__", "."): interpret_value(value)
        for name, value in env.items()
        if name.startswith(ENV_PREFIX) and name not in _CONTROL_ENV_VARS
    }

    result: dict = {}
    set(d, config=result, lock=threading.Lock())
    return result


def collect_legacy_env(
    env: Mapping[str, str] | None = None, defaults: list[Mapping] = defaults
) -> dict:
    """Collect config from deprecated ``DASK_``-prefixed environment variables

    Before abTEM read its own configuration location, ``refresh`` delegated
    wholesale to :func:`dask.config.collect`, so the only environment variables
    that reached abTEM were dask's. Those keep working for a transition period,
    but they warn, and only for keys abTEM itself defines -- a genuine dask
    setting such as ``DASK_DISTRIBUTED__WORKER__MEMORY__TARGET`` is left to
    dask and never enters abTEM's config.

    Parameters
    ----------
    env : Mapping[str, str], optional
        The system environment variables. Defaults to ``os.environ``.
    defaults : list of Mapping, optional
        The registered default configurations, used to decide which variables
        in dask's namespace are meant for abTEM.

    Returns
    -------
    config : dict
    """
    if env is None:
        env = os.environ

    d = {}
    for name, value in env.items():
        if not name.startswith(LEGACY_ENV_PREFIX):
            continue

        key = name[len(LEGACY_ENV_PREFIX) :].lower().replace("__", ".")

        if not _defines_key(key, defaults):
            continue

        new_name = ENV_PREFIX + name[len(LEGACY_ENV_PREFIX) :]
        warnings.warn(
            "Setting abTEM configuration through the environment variable "
            f'"{name}" is deprecated, as it collides with dask\'s '
            f'configuration namespace. Please use "{new_name}" instead.',
            FutureWarning,
            stacklevel=2,
        )
        d[key] = interpret_value(value)

    result: dict = {}
    set(d, config=result, lock=threading.Lock())
    return result


def collect(
    paths: list[str] = paths,
    env: Mapping[str, str] | None = None,
    defaults: list[Mapping] = defaults,
) -> dict:
    """
    Collect configuration from paths and environment variables

    Parameters
    ----------
    paths : list[str]
        A list of paths to search for yaml config files. Defaults to
        ``abtem.config.paths``, i.e. ``/etc/abtem``, ``<sys.prefix>/etc/abtem``,
        ``~/.config/abtem`` and ``$ABTEM_CONFIG``, in increasing priority.
    env : Mapping[str, str], optional
        The system environment variables. Defaults to ``os.environ``.
    defaults : list of Mapping, optional
        The registered default configurations, used by
        :func:`collect_legacy_env`.

    Returns
    -------
    config : dict

    See Also
    --------
    abtem.config.refresh: collect configuration and update into primary config
    """
    try:
        yaml_configs = list(collect_yaml(paths=paths))
    except ValueError as e:
        # A malformed yaml file in the user's config directory must not take
        # down `import abtem` -- a typo should degrade to a warning, not a raw
        # parser traceback. Coarse-grained: this drops every yaml source for
        # this call rather than isolating just the one bad file among several
        # in `paths`; real per-file isolation would mean replicating
        # collect_yaml's own path-discovery loop.
        warnings.warn(
            f"Failed to read abTEM's yaml configuration files under {paths}; "
            f"configuration from these files is being skipped: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        yaml_configs = []

    configs = [
        *yaml_configs,
        collect_legacy_env(env=env, defaults=defaults),
        collect_env(env=env),
    ]
    return merge(*configs)


def refresh(
    config: dict = config, defaults: list[Mapping] = defaults, **kwargs
) -> None:
    """
    Update configuration by re-reading yaml files and env variables

    This mutates the global abtem.config.config, or the config parameter if
    passed in.

    This goes through the following stages:

    1.  Clearing out all old configuration
    2.  Updating from the stored defaults from downstream libraries
        (see update_defaults)
    3.  Updating from abTEM's yaml files (see ``abtem.config.paths``) and
        ``ABTEM_``-prefixed environment variables

    Note that some functionality only checks configuration once at startup and
    may not change behavior, even if configuration changes.  It is recommended
    to restart your python process if convenient to ensure that new
    configuration changes take place.

    See Also
    --------
    abtem.config.collect: for parameters
    abtem.config.update_defaults
    """
    config.clear()

    for d in defaults:
        update(config, d, priority="old")

    kwargs.setdefault("defaults", defaults)

    update(config, collect(**kwargs))


def get(
    key: str,
    default: Any = no_default,
    config: dict = config,
    override_with: Any = None,
) -> Any:
    """
    Get elements from global config

    If ``override_with`` is not None this value will be passed straight back.
    Useful for getting kwarg defaults from abtek config.

    Use '.' for nested access
    """
    if override_with is not None:
        return override_with
    keys = key.split(".")
    result = config
    for k in keys:
        k = canonical_name(k, result)
        try:
            result = result[k]
        except (TypeError, IndexError, KeyError):
            if default is not no_default:
                return default
            else:
                raise
    return result


def update_defaults(
    new: Mapping, config: dict = config, defaults: list[Mapping] = defaults
) -> None:
    """Add a new set of defaults to the configuration

    It does two things:

    1.  Add the defaults to a global collection to be used by refresh later
    2.  Updates the global config with the new configuration
        prioritizing older values over newer ones
    """
    defaults.append(new)
    update(config, new, priority="old")


deprecations: dict[str, str | None] = {}


def check_deprecations(key: str, deprecations: dict = deprecations) -> str:
    """Check if the provided value has been renamed or removed

    Parameters
    ----------
    key : str
        The configuration key to check
    deprecations : Dict[str, str]
        The mapping of aliases

    Returns
    -------
    new: str
        The proper key, whether the original (if no deprecation) or the aliased
        value
    """
    if key in deprecations:
        new = deprecations[key]
        if new:
            warnings.warn(
                'Configuration key "{}" has been deprecated. '
                'Please use "{}" instead'.format(key, new)
            )
            return new
        else:
            raise ValueError(f'Configuration value "{key}" has been removed')
    else:
        return key


def _initialize() -> None:
    fn = os.path.join(os.path.dirname(__file__), "abtem.yaml")

    with open(fn, encoding="utf-8") as f:
        _defaults = yaml.safe_load(f)

    update_defaults(_defaults)


_initialize()
refresh()
