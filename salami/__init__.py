from salami.generator import AbstractGenerator
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pysalami")
except PackageNotFoundError:     
    __version__ = "0.0.0+unknown"

def default_threads():
    """Get the number of cpus available for parallel processing

    Returns:
        int: number of cpus
    """
    a = AbstractGenerator()
    return a.ncpus
