def convolve_probe(probe, atoms, shape, margin, intensities):
    extent = np.diag(atoms.cell)[:2]
    sampling = extent / np.array(shape)

    margin = int(np.ceil(margin / min(sampling)))
    shape_w_margin = (shape[0] + 2 * margin, shape[1] + 2 * margin)

    positions = atoms.positions[:, :2] / sampling

    inside = ((positions[:, 0] > -margin) &
              (positions[:, 1] > -margin) &
              (positions[:, 0] < shape[0] + margin) &
              (positions[:, 1] < shape[1] + margin))

    positions = positions[inside] + margin
    numbers = atoms.numbers[inside]

    if isinstance(intensities, float):
        intensities = {unique: unique ** intensities for unique in np.unique(numbers)}

    array = np.zeros((1,) + shape_w_margin)
    for number in np.unique(atoms.numbers):
        temp = np.zeros((1,) + shape_w_margin)
        superpose_deltas(positions[numbers == number], 0, temp)
        array += temp * intensities[number]

    probe = probe.copy()
    probe.extent = (shape_w_margin[0] * sampling[0], shape_w_margin[1] * sampling[1])
    probe.gpts = shape_w_margin
    intensity = probe.build((0, 0)).intensity()[0].array
    intensity /= intensity.max()

    array = np.fft.ifft2(np.fft.fft2(array) * np.fft.fft2(intensity)).real
    array = array[0, margin:-margin, margin:-margin]
    array = np.abs(array)

    calibrations = calibrations_from_grid(gpts=shape, sampling=sampling)
    return Measurement(array=array, calibrations=calibrations)
