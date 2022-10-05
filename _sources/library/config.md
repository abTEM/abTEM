# Configuration

You can configure *abTEM* with a YAML configuration file (abtem.yaml). This file controls a number of options and
feature flags.

## Configuration files

The configuration is specified by *any* YAML file in `~/.config/abtem/` or `/etc/abtem/`. *abTEM* searches for all YAML
files within each of these directories and merges them together.

Below is the full default configuration file. Anything you set in your own YAML will be merged into these
defaults before they are used to configure the build.

```{literalinclude} ./default_config.yaml
:language: yaml
:class: full-width
```

## Access configuration

```{eval-rst}
.. autosummary::
   abtem.config.get
```

*abTEM*’s configuration system is usually accessed using the `abtem.config.get` function. You can use `.` for nested
access, for example:

```python
import dask
dask.config.get("dask.chunk-size") # use `.` for nested access
```

## Specify configuration in Python

```{eval-rst}
.. autosummary::
   abtem.config.set
```

The configuration is stored within a normal Python dictionary in `abtem.config.config` and can be modified using normal
Python operations.

Additionally, you can temporarily set a configuration value using the `abtem.config.set` function. This function accepts
a dictionary as an input and interprets "." as nested access:

```python
abtem.config.set({"dask.chunk-size": "256 MB"})
```

This function can also be used as a context manager for consistent cleanup:

```python
with abtem.config.set({"dask.chunk-size": "256 MB"}):
    exit_waves = probe.multislice(potential)
```
