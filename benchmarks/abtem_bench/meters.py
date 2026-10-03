"""Meters used inside the worker process.

Every meter here is either a clock or a driver/allocator query. None of them
runs code on the computation being measured, and none touches dask workers.
Peak host memory is the worker's own ``VmHWM``, read once after the case has
finished: the high-water mark of the worker's address space since its exec.
``ru_maxrss`` from ``os.wait4`` is not used for it, because Linux carries the
parent's high-water mark into the child at fork and exec, which puts a floor
of the runner's own peak under every reading.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass, field


def _cupy():
    try:
        import cupy as cp  # type: ignore

        return cp
    except Exception:  # noqa: BLE001 -- cupy absent or unusable
        return None


def device_synchronize(device: str) -> None:
    if device != "gpu":
        return
    cp = _cupy()
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def gpu_cleanup(device: str) -> None:
    """Free pools between repeats so one repeat's allocations do not shadow the next."""
    gc.collect()
    if device != "gpu":
        return
    cp = _cupy()
    if cp is None:
        return
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@dataclass
class Timing:
    wall: float
    cpu: float


def timed(fn, device: str) -> tuple[object, Timing]:
    """Run ``fn()`` and return its result with wall and process-CPU seconds.

    On GPU the device is synchronised before and after so the wall time covers
    the device work, not just the host-side dispatch.
    """
    device_synchronize(device)
    t0 = time.perf_counter()
    c0 = time.process_time()
    result = fn()
    device_synchronize(device)
    return result, Timing(time.perf_counter() - t0, time.process_time() - c0)


@dataclass
class VRAMSampler:
    """Background thread sampling CuPy pool usage and driver-level device usage.

    ``peak_pool`` is ``used_bytes()`` of the default memory pool; ``peak_device``
    is ``total - free`` from ``cp.cuda.Device().mem_info``, which also captures
    cuFFT workspace held outside the pool. Both are queries, not computation.
    """

    device: str = "gpu"
    interval: float = 0.005
    peak_pool: int = 0
    peak_device: int = 0
    samples: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _active: bool = False

    def _loop(self, cp) -> None:
        try:
            pool = cp.get_default_memory_pool()
            dev = cp.cuda.Device()
        except Exception:  # noqa: BLE001 -- no usable device; record nothing
            return
        while not self._stop.is_set():
            try:
                used = pool.used_bytes()
                free, total = dev.mem_info
            except Exception:  # noqa: BLE001 -- device gone; stop sampling
                return
            self.peak_pool = max(self.peak_pool, used)
            self.peak_device = max(self.peak_device, total - free)
            self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "VRAMSampler":
        if self.device != "gpu":
            return self  # a CPU case must not initialise the GPU
        cp = _cupy()
        if cp is None:
            return self
        self._active = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(cp,), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def as_dict(self) -> dict:
        if not self._active or self.samples == 0:
            return {
                "peak_vram_pool_bytes": None,
                "peak_vram_device_bytes": None,
                "samples": 0,
            }
        return {
            "peak_vram_pool_bytes": int(self.peak_pool),
            "peak_vram_device_bytes": int(self.peak_device),
            "samples": self.samples,
        }


def vm_hwm_bytes() -> int | None:
    """This process's peak resident memory since exec (``VmHWM``), or None."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None
