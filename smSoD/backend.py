import numpy as np


def get_backend(name="auto"):
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
    return "numpy", np


def to_numpy(a):
    if hasattr(a, "get"):
        return a.get()
    return np.asarray(a)


class Rng:
    def __init__(self, xp, seed):
        self.is_np = xp is np
        if self.is_np:
            self.g = np.random.default_rng(seed)
        else:
            self.g = xp.random.RandomState(seed % (2 ** 32))

    def random(self, shape):
        if self.is_np:
            return self.g.random(shape)
        return self.g.random_sample(shape)

    def integers(self, high, shape):
        if self.is_np:
            return self.g.integers(0, high, size=shape)
        return self.g.randint(0, high, size=shape)
