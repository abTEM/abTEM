from collections import defaultdict

from abtem.core.axes import ThicknessSeriesAxis
from abtem.measure.detect import stack_measurements
import numpy as np


def thickness_series_precursor(detectors, potential):
    detect_every = defaultdict(list)
    all_detect_at = []
    axes_metadata = []
    for detector in detectors:
        if detector.detect_every:
            detect_every[detector.detect_every].append(detector)

            if not len(potential) == detector.detect_every:
                detect_every[len(potential)].append(detector)

            detect_at = list(range(0, len(potential), detector.detect_every))
            if not len(potential) % detector.detect_every:
                detect_at += [len(potential)]

            all_detect_at += detect_at

            multislice_start_stop = [(detect_at[i], detect_at[i + 1]) for i in range(len(detect_at) - 1)]
            domain = np.cumsum([potential.slice_thickness[start:stop].sum() for start, stop in multislice_start_stop])
            axes_metadata += [ThicknessSeriesAxis(values=tuple(domain))]
        else:
            detect_every[len(potential)].append(detector)
            axes_metadata += [ThicknessSeriesAxis(values=(potential.thickness,))]

    all_detect_at += [0, len(potential)]
    all_detect_at = sorted(list(set(all_detect_at)))
    multislice_start_stop = [(all_detect_at[i], all_detect_at[i + 1]) for i in range(len(all_detect_at) - 1)]
    return multislice_start_stop, detect_every, axes_metadata


def detectors_at_stop_slice(detect_every, stop_slice):
    detectors_at = []
    for n, detectors in detect_every.items():
        for detector in detectors:
            if stop_slice % n == 0 and detector not in detectors_at:
                detectors_at.append(detector)
    return detectors_at


def stack_thickness_series(measurements, detectors, axes_metadata):
    # measurements = list(map(list, zip(*measurements)))
    stacked = []
    for detector, metadata in zip(detectors, axes_metadata):
        stacked.append(stack_measurements(measurements[detector], axes_metadata=axes_metadata))
    return stacked
