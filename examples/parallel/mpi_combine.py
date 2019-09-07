import numpy as np
import os
import matplotlib.pyplot as plt

folder = 'export'

partitions = {}
for f in os.listdir(folder):
    key = f.split('.')[-2].split('-')[-2:]
    key = tuple([int(x) for x in key])
    partitions[key] = np.load(os.path.join(folder, f))

n = max([key[0] for key in partitions.keys()]) + 1
m = max([key[1] for key in partitions.keys()]) + 1

N = sum([partition.shape[0] for key, partition in partitions.items() if key[0] == 0])
M = sum([partition.shape[1] for key, partition in partitions.items() if key[1] == 0])

array = np.zeros((N, M))

k = 0
for i in range(n):
    l = 0
    for j in range(m):
        partition = partitions[(i, j)]
        array[l:l + partition.shape[0], k:k + partition.shape[1]] = partition
        l += partition.shape[0]
    k += partition.shape[1]

plt.imshow(array)
plt.show()