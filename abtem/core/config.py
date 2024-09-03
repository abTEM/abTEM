import os
import threading
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Union

import yaml
from dask.config import canonical_name, collect, update

no_default = "__no_default__"


if "ABTEM_CONFIG" in os.environ:
    PATH = os.environ["ABTEM_CONFIG"]
else:
    PATH = os.path.join(os.path.expanduser("~"), ".config", "abtem")

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
    3.  Updating from yaml files and environment variables

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

    Examples
    --------
    >>> deprecations = {"old_key": "new_key", "invalid": None}
    >>> check_deprecations("old_key", deprecations=deprecations)  # doctest: +SKIP
    UserWarning: Configuration key "old_key" has been deprecated. Please use "new_key" instead.

    >>> check_deprecations("invalid", deprecations=deprecations)
    Traceback (most recent call last):
        ...
    ValueError: Configuration value "invalid" has been removed

    >>> check_deprecations("another_key", deprecations=deprecations)
    'another_key'

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

    with open(fn) as f:
        _defaults = yaml.safe_load(f)

    update_defaults(_defaults)


refresh()
_initialize()
