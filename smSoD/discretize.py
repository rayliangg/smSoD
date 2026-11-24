import numpy as np

N_TRIAL = 5
DENSE_LIMIT = 1 << 22


class Codebook:
    def __init__(self, codes, local, lo, hi, sizes):
        self.codes = codes    # (m, N) leaf id
        self.local = local    # (m, N) leaf id inside each subproblem
        self.lo = lo
        self.hi = hi
        self.sizes = sizes    # leaves per subproblem


class Discretizer:
    def __init__(self, cfg, rng, gpu=None):
        self.cfg = cfg
        self.rng = rng
        self.gpu = gpu
        self.gpu_calls = 0
        k = cfg.k
        self.K = 1 << k
        self.pow2 = np.left_shift(1, np.arange(k, dtype=np.int64))
        self.pow2f = self.pow2.astype(np.float64)
        self.bit = np.arange(k, dtype=np.int64)
        self.trial_id = np.arange(N_TRIAL, dtype=np.int64)
        if gpu is not None:
            self.d_pow2f = gpu.asarray(self.pow2f)
            self.d_trial_id = gpu.asarray(self.trial_id)

    def pack(self, bits):
        if self.cfg.k <= 52:
            return (bits.astype(np.float64) @ self.pow2f).astype(np.int64)
        return (bits * self.pow2).sum(axis=-1)

    def trial_score(self, X, trial, reg, R):
        # for every (region, trial): size of the most crowded orthant
        n, k = X.shape
        G = R * N_TRIAL
        if (self.gpu is not None and n * N_TRIAL * k >= self.cfg.gpu_min
                and G * self.K <= DENSE_LIMIT and k <= 52):
            return self._trial_score_gpu(X, trial, reg, G)

        mask = self.pack(X[:, None, :] > trial[reg])
        group = reg[:, None] * N_TRIAL + self.trial_id
        if G * self.K <= DENSE_LIMIT:
            cnt = np.bincount((group * self.K + mask).ravel(), minlength=G * self.K)
            return cnt.reshape(G, self.K).max(axis=1)

        _, inv = np.unique(mask.ravel(), return_inverse=True)
        inv = inv.ravel()
        U = int(inv.max()) + 1
        key, cnt = np.unique(group.ravel() * U + inv, return_counts=True)
        out = np.zeros(G, dtype=np.int64)
        np.maximum.at(out, key // U, cnt)
        return out

    def _trial_score_gpu(self, X, trial, reg, G):
        cp = self.gpu
        self.gpu_calls += 1
        dX = cp.asarray(X)
        dt = cp.asarray(trial)
        dreg = cp.asarray(reg)
        bits = (dX[:, None, :] > dt[dreg]).astype(cp.float64)
        mask = (bits @ self.d_pow2f).astype(cp.int64)
        key = (dreg[:, None] * N_TRIAL + self.d_trial_id) * self.K + mask
        cnt = cp.bincount(key.ravel(), minlength=G * self.K)
        return cnt.reshape(G, self.K).max(axis=1).get()

    def group_children(self, region, mask, n_region):
        # points with the same (region, orthant) go to the same child
        if n_region * self.K <= DENSE_LIMIT:
            key = region * self.K + mask
            cnt = np.bincount(key, minlength=n_region * self.K)
            used = np.nonzero(cnt)[0]
            lut = np.empty(n_region * self.K, dtype=np.int64)
            lut[used] = np.arange(used.size)
            return used // self.K, used % self.K, cnt[used], lut[key]

        u, inv = np.unique(mask, return_inverse=True)
        U = u.size
        key, cinv, cnt = np.unique(region * U + inv.ravel(), return_inverse=True, return_counts=True)
        return key // U, u[key % U], cnt, cinv.ravel()

    def __call__(self, points, split_rate, min_bound):
        cfg = self.cfg
        m, N, k = points.shape
        max_points = int(cfg.pop_size * split_rate)

        pts = points.reshape(m * N, k)
        codes = np.empty(m * N, dtype=np.int64)

        # split all regions of all subproblems level by level
        pid = np.arange(m * N, dtype=np.int64)
        reg = pid // N
        lo = np.full((m, k), cfg.lower)
        hi = np.full((m, k), cfg.upper)
        sub = np.arange(m, dtype=np.int64)

        leaf_lo, leaf_hi, leaf_sub = [], [], []
        n_leaf = 0

        while pid.size > 0:
            R = lo.shape[0]
            X = pts[pid]

            # find best split point
            trial = lo[:, None, :] + (hi - lo)[:, None, :] * self.rng.random((R, N_TRIAL, k))
            score = self.trial_score(X, trial, reg, R)
            pick = np.argmax(score.reshape(R, N_TRIAL), axis=1)
            small = np.bincount(reg, minlength=R) <= cfg.tour_size
            pick[small] = 0
            sp = trial[np.arange(R), pick]

            parent, orth, ccnt, child = self.group_children(reg, self.pack(X > sp[reg]), R)
            up = ((orth[:, None] >> self.bit) & 1).astype(bool)
            p_sp = sp[parent]
            clo = np.where(up, p_sp, lo[parent])
            chi = np.where(up, hi[parent], p_sp)
            csub = sub[parent]

            go_on = (ccnt >= max_points) & ((chi - clo) >= min_bound).all(axis=1)
            is_leaf = ~go_on
            leaf_id = np.cumsum(is_leaf) - 1 + n_leaf
            done = is_leaf[child]
            codes[pid[done]] = leaf_id[child[done]]
            leaf_lo.append(clo[is_leaf])
            leaf_hi.append(chi[is_leaf])
            leaf_sub.append(csub[is_leaf])
            n_leaf += int(is_leaf.sum())

            keep = ~done
            pid = pid[keep]
            reg = (np.cumsum(go_on) - 1)[child[keep]]
            lo, hi, sub = clo[go_on], chi[go_on], csub[go_on]

        leaf_sub = np.concatenate(leaf_sub)

        order = np.argsort(leaf_sub, kind="stable")
        sizes = np.bincount(leaf_sub, minlength=m)
        start = np.cumsum(sizes) - sizes
        local_id = np.empty(n_leaf, dtype=np.int64)
        local_id[order] = np.arange(n_leaf) - start[leaf_sub[order]]

        codes = codes.reshape(m, N)
        return Codebook(codes, local_id[codes], np.concatenate(leaf_lo),
                        np.concatenate(leaf_hi), sizes)
