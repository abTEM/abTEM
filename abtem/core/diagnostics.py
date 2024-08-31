from __future__ import annotations

import warnings
from typing import Any, Optional

from abtem.core import config

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:
    tqdm = None


class TqdmWrapper:
    """
    This class is a wrapper for the tqdm bar, which implements fallback logic if tqdm is not installed.

    Initializes TqdmWrapper with user_tqdm flag, total iterations, and additional arguments for tqdm.

    Parameters
    ----------
    enabled : bool, optional
        A flag indicating if the wrapper is enabled. If None, the value from the configuration key
        "local_diagnostics.task_level_progress" is used.
    *args
        Variable length argument list for tqdm.
    **kwargs
        Arbitrary keyword arguments for tqdm.

    Raises
    ------
    Warning
        Issues a warning if the progress display is enabled but tqdm is not installed.
    """

    def __init__(self, *args, enabled: Optional[bool] = None, **kwargs: Any):
        if enabled is None:
            enabled = config.get("local_diagnostics.task_level_progress", False)

        if tqdm is not None and enabled:
            self._pbar = tqdm(*args, **kwargs)
        else:
            if enabled:
                warnings.warn("displaying task level progress require tqdm installed")

            self._pbar = None

    @property
    def pbar(self):
        """The progress bar object."""
        return self._pbar

    def update_if_exists(self, n: int = 1) -> None:
        """
        Updates the progress bar by n steps, if tqdm is successfully imported and enabled.

        Parameters
        ----------
        n : int, optional
            The number of steps by which to increment the progress bar.
        """
        if self.pbar is not None:
            self.pbar.update(n)

    def close_if_exists(self) -> None:
        """
        Closes the progress bar provided it exists.
        """
        if self.pbar is not None:
            self.pbar.close()
