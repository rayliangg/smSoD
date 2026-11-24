class RidgeProblem:
    # f = sum_i (x_i0 - t_i)^2 + sum_j (x_ij - x_i,j-1)^2, optimum = 0
    def __init__(self, cfg, xp, rng):
        self.m = cfg.m
        self.k = cfg.k
        self.targets = cfg.lower + (cfg.upper - cfg.lower) * rng.random(cfg.m)

    def __call__(self, genes):
        g = genes.reshape(-1, self.m, self.k)
        d0 = g[:, :, 0] - self.targets
        f = (d0 * d0).sum(axis=1)
        if self.k > 1:
            d = g[:, :, 1:] - g[:, :, :-1]
            f += (d * d).sum(axis=(1, 2))
        return f
