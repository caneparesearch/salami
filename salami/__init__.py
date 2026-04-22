from salami.generator import AbstractGenerator


def default_threads():
    """Get the number of cpus available for parallel processing

    Returns:
        int: number of cpus
    """
    a = AbstractGenerator()
    return a.ncpus
