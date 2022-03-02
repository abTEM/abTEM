# def remove_tilt(self):
#     xp = get_array_module(self.array)
#     if self.is_lazy:
#         array = self.array.map_blocks(remove_tilt,
#                                       planewave_cutoff=self.planewave_cutoff,
#                                       extent=self.extent,
#                                       gpts=self.gpts,
#                                       energy=self.energy,
#                                       interpolation=self.interpolation,
#                                       partitions=self.partitions,
#                                       accumulated_defocus=self.accumulated_defocus,
#                                       meta=xp.array((), dtype=xp.complex64))
#     else:
#         array = remove_tilt(self.array,
#                             planewave_cutoff=self.planewave_cutoff,
#                             extent=self.extent,
#                             gpts=self.gpts,
#                             energy=self.energy,
#                             interpolation=self.interpolation,
#                             partitions=self.partitions,
#                             accumulated_defocus=self.accumulated_defocus)
#
#     self._array = array
#     return self
#
#
# def interpolate_full(self, chunks):
#     xp = get_array_module(self.array)
#     self.remove_tilt()
#     self.rechunk()
#
#     wave_vectors = prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy, self.interpolation)
#
#     arrays = []
#     for start, end in generate_chunks(len(wave_vectors), chunks=chunks):
#         array = dask.delayed(interpolate_full)(array=self.array,
#                                                parent_wave_vectors=self.wave_vectors,
#                                                wave_vectors=wave_vectors[start:end],
#                                                extent=self.extent,
#                                                gpts=self.gpts,
#                                                energy=self.energy,
#                                                defocus=self.accumulated_defocus)
#
#         array = da.from_delayed(array, shape=(end - start,) + self.gpts, dtype=xp.complex64)
#         array = array * np.sqrt(len(self) / len(wave_vectors))
#         arrays.append(array)
#
#     array = da.concatenate(arrays)
#     d = self._copy_as_dict(copy_array=False)
#     d['array'] = array
#     d['wave_vectors'] = wave_vectors
#     d['partitions'] = None
#     return self.__class__(**d)
#
#
# extent = (self.interpolated_gpts[0] * self.sampling[0],
#           self.interpolated_gpts[1] * self.sampling[1])
#
# wave_vectors = prism_wave_vectors(self.planewave_cutoff, extent, self.energy, (1, 1))
#
# ctf = ctf.copy()
# ctf.defocus = -self.accumulated_defocus
#
# basis = beamlet_basis(ctf,
#                       self.wave_vectors,
#                       wave_vectors,
#                       self.interpolated_gpts,
#                       self.sampling).astype(np.complex64)
#
# else:
# wave_vectors = partitioned_prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy,
#                                               self.partitions, num_points_per_ring=6, xp=xp)
#
#
#
#
# def _reduce_partitioned(s_matrix, basis, positions: np.ndarray, axes_metadata) -> Waves:
#     if len(axes_metadata) != (len(positions.shape) - 1):
#         raise RuntimeError()
#
#     shifts = np.round(positions.reshape((-1, 2)) / s_matrix.sampling).astype(int)
#     shifts -= np.array(s_matrix.crop_offset)
#     shifts -= (np.array(s_matrix.interpolated_gpts)) // 2
#
#     # basis = np.moveaxis(basis, 0, 2).copy()
#     # array = np.moveaxis(s_matrix.array, 0, 2).copy()
#
#     import warnings
#     warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
#
#     waves = np.zeros((len(shifts),) + s_matrix.interpolated_gpts, dtype=np.complex64)
#     reduce_beamlets_nearest_no_interpolation(waves, basis, s_matrix.array, shifts)
#     waves = waves.reshape(positions.shape[:-1] + waves.shape[-2:])
#
#     waves = Waves(waves,
#                   sampling=s_matrix.sampling,
#                   energy=s_matrix.energy,
#                   extra_axes_metadata=axes_metadata,
#                   antialias_cutoff_gpts=s_matrix.antialias_cutoff_gpts)
#
#     return waves
#
# # if self.partitions:
# #     extent = (self.interpolated_gpts[0] * self.sampling[0],
# #               self.interpolated_gpts[1] * self.sampling[1])
# #
# #     wave_vectors = prism_wave_vectors(self.planewave_cutoff, extent, self.energy, (1, 1))
# #
# #     ctf = ctf.copy()
# #     ctf.defocus = -self.accumulated_defocus
# #
# #     basis = beamlet_basis(ctf,
# #                           self.wave_vectors,
# #                           wave_vectors,
# #                           self.interpolated_gpts,
# #                           self.sampling).astype(np.complex64)
#
#
# def linear_scaling_transition_scan(self, scan, collection_angle, transitions, ctf: CTF = None,
#                                    reverse_multislice=False, lazy=False):
#     d = self._copy_as_dict(copy_potential=False)
#     d['potential'] = self.potential
#     d['planewave_cutoff'] = collection_angle
#     S2 = self.__class__(**d)
#
#     ctf = self._validate_ctf(ctf)
#     scan = self._validate_positions(positions=scan, ctf=ctf)
#
#     if hasattr(transitions, 'get_transition_potentials'):
#         if lazy:
#             transitions = dask.delayed(transitions.get_transition_potentials)()
#         else:
#             transitions = transitions.get_transition_potentials()
#
#     return linear_scaling_transition_multislice(self, S2, scan, transitions, reverse_multislice=reverse_multislice)
