Measurement
===========

.. currentmodule:: abtem.measurements.core

.. autoclass:: Measurement
   :members:
   :show-inheritance:
   :inherited-members:

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~Measurement.__init__
      ~Measurement.check_axes_metadata
      ~Measurement.check_is_compatible
      ~Measurement.compute
      ~Measurement.copy
      ~Measurement.copy_kwargs
      ~Measurement.copy_to_device
      ~Measurement.ensure_lazy
      ~Measurement.expand_dims
      ~Measurement.find_axes_type
      ~Measurement.from_array_and_metadata
      ~Measurement.from_zarr
      ~Measurement.get_items
      ~Measurement.iterate_ensemble
      ~Measurement.max
      ~Measurement.mean
      ~Measurement.min
      ~Measurement.poisson_noise
      ~Measurement.power
      ~Measurement.rechunk
      ~Measurement.reduce_ensemble
      ~Measurement.relative_difference
      ~Measurement.scan_extent
      ~Measurement.scan_positions
      ~Measurement.squeeze
      ~Measurement.std
      ~Measurement.sum
      ~Measurement.to_cpu
      ~Measurement.to_delayed
      ~Measurement.to_gpu
      ~Measurement.to_hyperspy
      ~Measurement.to_zarr
      ~Measurement.visualize_graph
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Measurement.array
      ~Measurement.axes_metadata
      ~Measurement.base_axes
      ~Measurement.base_axes_metadata
      ~Measurement.base_shape
      ~Measurement.chunks
      ~Measurement.device
      ~Measurement.dimensions
      ~Measurement.dtype
      ~Measurement.energy
      ~Measurement.ensemble_axes
      ~Measurement.ensemble_axes_metadata
      ~Measurement.ensemble_shape
      ~Measurement.is_complex
      ~Measurement.is_lazy
      ~Measurement.metadata
      ~Measurement.num_axes
      ~Measurement.num_base_axes
      ~Measurement.num_ensemble_axes
      ~Measurement.num_scan_axes
      ~Measurement.scan_axes
      ~Measurement.scan_axes_metadata
      ~Measurement.scan_sampling
      ~Measurement.scan_shape
      ~Measurement.shape
      ~Measurement.wavelength
   
   