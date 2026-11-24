import numpy as np


def tournament(fitness, s, rng):
    N = fitness.shape[0]
    per_shuffle = N // s
    n_shuffle = -(-N // per_shuffle)

    perm = np.argsort(rng.random((n_shuffle, N)), axis=1)
    groups = perm[:, :per_shuffle * s].reshape(-1, s)[:N]
    best = np.argmin(fitness[groups], axis=1)
    return groups[np.arange(N), best]
