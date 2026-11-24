import numpy as np


def sample_offspring(book, n_child, groups, rng, per_group=False):
    m, N = book.codes.shape

    if per_group:
        g_of = np.empty(m, dtype=np.int64)
        for gi, mem in enumerate(groups):
            g_of[mem] = gi
        parent = rng.integers(0, N, size=(n_child, len(groups)))[:, g_of]
    else:
        parent = rng.integers(0, N, size=(n_child, m))

    leaf = book.codes[np.arange(m), parent]
    lo = book.lo[leaf]
    hi = book.hi[leaf]
    child = lo + (hi - lo) * rng.random(lo.shape)
    return child.reshape(n_child, -1)
