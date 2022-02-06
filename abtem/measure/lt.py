import numpy as np
from libertem.api import Context
from libertem.common.container import MaskContainer
from libertem.corrections.coordinates import identity
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.memory import MemoryDataSet
from ptychography40.reconstruction.ssb import SSB_UDF, generate_masks

from abtem.measure.measure import Images


def make_ssb_params(measurements, cutoff=1., cutoff_freq=np.inf, method='shift'):
    rec_params = {
        'dtype': np.float32,
        'lamb': measurements.wavelength * 1e-10,
        'dpix': (measurements.scan_sampling[0] * 1e-10, measurements.scan_sampling[1] * 1e-10),
        'semiconv': measurements.metadata['semiangle_cutoff'] * 1e-3,
        'semiconv_pix': np.round(
            measurements.metadata['semiangle_cutoff'] / np.mean(measurements.angular_sampling)).astype(int),
        'transformation': identity(),
        'cy': measurements.array.shape[-2] // 2,
        'cx': measurements.array.shape[-1] // 2,
        'cutoff': cutoff,

    }

    mask_params = {
        'reconstruct_shape': measurements.scan_shape,
        'mask_shape': measurements.base_shape,
        'cutoff_freq': cutoff_freq,
        'method': method,
    }

    return rec_params, mask_params


def run_ssb(measurement, cutoff=1.):
    array = np.squeeze(measurement.array)
    rec_params, mask_params = make_ssb_params(measurement, cutoff=cutoff)
    ds = MemoryDataSet(data=array)
    ctx = Context(executor=InlineJobExecutor())
    trotters = generate_masks(**rec_params, **mask_params)
    mask_container = MaskContainer(mask_factories=lambda: trotters, dtype=trotters.dtype, count=trotters.shape[0])
    udf = SSB_UDF(**rec_params, mask_container=mask_container)
    udf_result = ctx.run_udf(udf=udf, dataset=ds, progress=False)
    return Images(udf_result['phase'].data[None], sampling=measurement.scan_sampling,
                  extra_axes_metadata=measurement.extra_axes_metadata[:-2])
