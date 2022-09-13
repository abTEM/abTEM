Waves
=====

.. currentmodule:: abtem.waves.core

.. autoclass:: Waves
   :members:
   :show-inheritance:
   :inherited-members:

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~Waves.__init__
      ~Waves.apply_ctf
      ~Waves.apply_transform
      ~Waves.as_complex_diffraction
      ~Waves.as_complex_image
      ~Waves.check_axes_metadata
      ~Waves.check_is_compatible
      ~Waves.complex_images
      ~Waves.compute
      ~Waves.convolve
      ~Waves.copy
      ~Waves.copy_kwargs
      ~Waves.copy_to_device
      ~Waves.diffraction_patterns
      ~Waves.downsample
      ~Waves.ensure_fourier_space
      ~Waves.ensure_lazy
      ~Waves.ensure_real_space
      ~Waves.expand_dims
      ~Waves.find_axes_type
      ~Waves.fresnel_propagator
      ~Waves.from_array_and_metadata
      ~Waves.from_partitioned_args
      ~Waves.from_zarr
      ~Waves.get_items
      ~Waves.intensity
      ~Waves.match_grid
      ~Waves.max
      ~Waves.mean
      ~Waves.min
      ~Waves.multislice
      ~Waves.phase_shift
      ~Waves.rechunk
      ~Waves.renormalize
      ~Waves.show
      ~Waves.squeeze
      ~Waves.std
      ~Waves.sum
      ~Waves.tile
      ~Waves.to_cpu
      ~Waves.to_delayed
      ~Waves.to_gpu
      ~Waves.to_zarr
      ~Waves.visualize_graph
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Waves.accelerator
      ~Waves.angular_sampling
      ~Waves.antialias_cutoff_gpts
      ~Waves.antialias_valid_gpts
      ~Waves.array
      ~Waves.axes_metadata
      ~Waves.base_axes
      ~Waves.base_axes_metadata
      ~Waves.base_shape
      ~Waves.base_tilt
      ~Waves.chunks
      ~Waves.cutoff_angles
      ~Waves.device
      ~Waves.dtype
      ~Waves.energy
      ~Waves.ensemble_axes
      ~Waves.ensemble_axes_metadata
      ~Waves.ensemble_shape
      ~Waves.extent
      ~Waves.fourier_space
      ~Waves.fourier_space_axes_metadata
      ~Waves.fourier_space_sampling
      ~Waves.full_cutoff_angles
      ~Waves.gpts
      ~Waves.grid
      ~Waves.is_complex
      ~Waves.is_lazy
      ~Waves.metadata
      ~Waves.num_axes
      ~Waves.num_base_axes
      ~Waves.num_ensemble_axes
      ~Waves.num_scan_axes
      ~Waves.rectangle_cutoff_angles
      ~Waves.sampling
      ~Waves.scan_axes
      ~Waves.scan_axes_metadata
      ~Waves.scan_sampling
      ~Waves.scan_shape
      ~Waves.shape
      ~Waves.tilt_axes
      ~Waves.tilt_axes_metadata
      ~Waves.wavelength
   
   