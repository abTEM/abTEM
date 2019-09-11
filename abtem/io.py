import os
from glob import glob
import numpy as np


def decode_partition(f):
    position = f.split('_')[-1].split('.')[0]
    shape = tuple(map(int, position.split('in')[-1].split('x')))
    start, stop = map(int, position.split('in')[0].split('-'))
    return start, stop, shape


def import_partition(template):
    partition_files = glob(template)

    pattern_shape = np.load(os.path.join(partition_files[0])).shape[1:]
    _, _, partition_shape = decode_partition(partition_files[0])

    partition = np.zeros((np.prod(partition_shape),) + pattern_shape)

    for f in partition_files:
        start, stop, _ = decode_partition(f)
        partition[start:stop] = np.load(f)

    partition = partition.reshape(partition_shape + pattern_shape)
    return partition


def import_partitions(folder):
    files = os.listdir(folder)

    n = max([int(f.split('_')[-2].split('-')[-2]) for f in files]) + 1
    m = max([int(f.split('_')[-2].split('-')[-1]) for f in files]) + 1

    partitions = {}
    for i in range(n):
        for j in range(m):
            template = os.path.join(folder, 'partition-{}-{}'.format(i, i)) + '*'
            partitions[(i, j)] = import_partition(template)
    return partitions


def assemble_partitions(partitions):
    n = max([key[0] for key in partitions.keys()]) + 1
    m = max([key[1] for key in partitions.keys()]) + 1

    N = sum([partition.shape[0] for key, partition in partitions.items() if key[1] == 0])
    M = sum([partition.shape[1] for key, partition in partitions.items() if key[0] == 0])

    data = np.zeros((N, M) + partitions[(0, 0)].shape[-2:])

    l = 0
    for i in range(n):
        k = 0
        for j in range(m):
            partition = partitions[(i, j)]
            data[l:l + partition.shape[0], k:k + partition.shape[1]] = partition
            k += partition.shape[1]
        l += partition.shape[0]
    return data


def import_ptychography_data(folder):
    return assemble_partitions(import_partitions(folder))
