import numpy as np


# returns (name, cupy module) or ("numpy", None)
def get_backend(name="auto"):
    if name not in ("auto", "numpy", "cupy"):
        raise ValueError("unknown backend: %s" % name)

    if name in ("auto", "cupy"):
        try:
            import cupy as cp
            (cp.arange(4) * 2.0).sum().item()
            return "cupy", cp
        except Exception as e:
            if name == "cupy":
                raise RuntimeError("CuPy backend unavailable: %s" % e)
    return "numpy", None

