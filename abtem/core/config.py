from dask.config import *

paths = [
    os.getenv("ABTEM_ROOT_CONFIG", "/etc/abtem"),
    os.path.join(sys.prefix, "etc", "abtem"),
    os.path.join(os.path.expanduser("~"), ".config", "abtem"),
    os.path.join(os.path.expanduser("~"), ".abtem"),
]

if "ABTEM_CONFIG" in os.environ:
    PATH = os.environ["ABTEM_CONFIG"]
    paths.append(PATH)
else:
    PATH = os.path.join(os.path.expanduser("~"), ".config", "abtem")


def _initialize():
    fn = os.path.join(os.path.dirname(__file__), "abtem.yaml")

    with open(fn) as f:
        _defaults = yaml.safe_load(f)

    update_defaults(_defaults)


refresh()
_initialize()
