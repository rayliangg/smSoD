def tournament(fitness, s, rng, xp):
    N = fitness.shape[0]
    per_shuffle = N // s
    n_shuffle = -(-N // per_shuffle)

    perm = xp.argsort(rng.random((n_shuffle, N)), axis=1)
    groups = perm[:, :per_shuffle * s].reshape(-1, s)[:N]
    best = xp.argmin(fitness[groups], axis=1)
    return groups[xp.arange(N), best]
