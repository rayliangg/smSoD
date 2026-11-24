import numpy as np


def get_backend(name="auto"):
    """returns (name, gpu). gpu is the cupy module, or None for cpu only"""
    if name not in ("auto", "numpy", "cupy"):
        raise ValueError("unknown backend: %s" % name)

    if name in ("auto", "cupy"):
        try:
            import cupy as cp
            (cp.random.RandomState(0).random_sample(4) * 2.0).sum().item()
            return "cupy", cp
        except Exception as e:
            if name == "cupy":
                raise RuntimeError("CuPy backend unavailable: %s" % e)
    return "numpy", None

