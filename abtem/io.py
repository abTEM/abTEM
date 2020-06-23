import h5py


def abtem_read(path):
    with h5py.File(path, 'r') as f:
        datasets = {}
        for key in f.keys():
            datasets[key] = f.get(key)[()]

    klass = datasets.pop('class').item().decode('utf-8')
    mod = __import__('.'.join(klass.split('.')[:-1]), fromlist=[klass.split('.')[-1]])
    return getattr(mod, klass.split('.')[-1])(**datasets)