from dask.utils import parse_bytes

from abtem.core.backend import cp
from abtem.core import config


def initialize():
    if cp is not None:
        from cupy.fft.config import get_plan_cache

        cache = get_plan_cache()
        cache_size = parse_bytes(config.get('cupy.fft-cache-size'))
        cache.set_size(cache_size)
