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
    def __init__(self, cfg, xp, rng):
        self.cfg = cfg
        self.xp = xp
        self.rng = rng
        k = cfg.k
        self.K = 1 << k
        self.pow2 = xp.left_shift(1, xp.arange(k, dtype=xp.int64))
        self.pow2f = self.pow2.astype(xp.float64)
        self.bit = xp.arange(k, dtype=xp.int64)
        self.trial_id = xp.arange(N_TRIAL, dtype=xp.int64)

    def pack(self, bits):
        xp = self.xp
        if self.cfg.k <= 52:
            return (bits.astype(xp.float64) @ self.pow2f).astype(xp.int64)
        return (bits * self.pow2).sum(axis=-1)

    def max_orthant_count(self, group, mask, n_group):
        xp = self.xp
        if n_group * self.K <= DENSE_LIMIT:
            cnt = xp.bincount((group * self.K + mask).ravel(), minlength=n_group * self.K)
            return cnt.reshape(n_group, self.K).max(axis=1)

        _, inv = xp.unique(mask.ravel(), return_inverse=True)
        U = int(inv.max()) + 1
        key, cnt = xp.unique(group.ravel() * U + inv.ravel(), return_counts=True)
        B = self.cfg.pop_size + 1
        v = xp.sort((key // U) * B + cnt)
        g = v // B
        last = xp.ones(v.shape, dtype=bool)
        last[:-1] = g[:-1] != g[1:]
        out = xp.zeros(n_group, dtype=xp.int64)
        out[g[last]] = v[last] % B
        return out

    def group_children(self, region, mask, n_region):
        xp = self.xp
        if n_region * self.K <= DENSE_LIMIT:
            key = region * self.K + mask
            cnt = xp.bincount(key, minlength=n_region * self.K)
            used = xp.nonzero(cnt)[0]
            lut = xp.empty(n_region * self.K, dtype=xp.int64)
            lut[used] = xp.arange(used.size)
            return used // self.K, used % self.K, cnt[used], lut[key]

        u, inv = xp.unique(mask, return_inverse=True)
        U = u.size
        key, cinv, cnt = xp.unique(region * U + inv.ravel(), return_inverse=True, return_counts=True)
        return key // U, u[key % U], cnt, cinv.ravel()

    def __call__(self, points, split_rate, min_bound):
        xp = self.xp
        cfg = self.cfg
        m, N, k = points.shape
        max_points = int(cfg.pop_size * split_rate)

        pts = points.reshape(m * N, k)
        codes = xp.empty(m * N, dtype=xp.int64)

        # split all regions of all subproblems level by level
        pid = xp.arange(m * N, dtype=xp.int64)
        reg = pid // N
        lo = xp.full((m, k), cfg.lower)
        hi = xp.full((m, k), cfg.upper)
        sub = xp.arange(m, dtype=xp.int64)

        leaf_lo, leaf_hi, leaf_sub = [], [], []
        n_leaf = 0

        while pid.size > 0:
            R = lo.shape[0]
            X = pts[pid]

            # find best split point
            trial = lo[:, None, :] + (hi - lo)[:, None, :] * self.rng.random((R, N_TRIAL, k))
            mask = self.pack(X[:, None, :] > trial[reg])
            score = self.max_orthant_count(reg[:, None] * N_TRIAL + self.trial_id, mask, R * N_TRIAL)
            pick = xp.argmax(score.reshape(R, N_TRIAL), axis=1)
            small = xp.bincount(reg, minlength=R) <= cfg.tour_size
            pick = xp.where(small, 0, pick)
            sp = trial[xp.arange(R), pick]

            parent, orth, ccnt, child = self.group_children(reg, self.pack(X > sp[reg]), R)
            up = ((orth[:, None] >> self.bit) & 1).astype(bool)
            p_sp = sp[parent]
            clo = xp.where(up, p_sp, lo[parent])
            chi = xp.where(up, hi[parent], p_sp)
            csub = sub[parent]

            go_on = (ccnt >= max_points) & ((chi - clo) >= min_bound).all(axis=1)
            is_leaf = ~go_on
            leaf_id = xp.cumsum(is_leaf) - 1 + n_leaf
            done = is_leaf[child]
            codes[pid[done]] = leaf_id[child[done]]
            leaf_lo.append(clo[is_leaf])
            leaf_hi.append(chi[is_leaf])
            leaf_sub.append(csub[is_leaf])
            n_leaf += int(is_leaf.sum())

            keep = ~done
            pid = pid[keep]
            reg = (xp.cumsum(go_on) - 1)[child[keep]]
            lo, hi, sub = clo[go_on], chi[go_on], csub[go_on]

        leaf_sub = xp.concatenate(leaf_sub)

        order = xp.argsort(leaf_sub, kind="stable")
        sizes = xp.bincount(leaf_sub, minlength=m)
        start = xp.cumsum(sizes) - sizes
        local_id = xp.empty(n_leaf, dtype=xp.int64)
        local_id[order] = xp.arange(n_leaf) - start[leaf_sub[order]]

        codes = codes.reshape(m, N)
        sizes = sizes.get() if hasattr(sizes, "get") else sizes
        return Codebook(codes, local_id[codes], xp.concatenate(leaf_lo),
                        xp.concatenate(leaf_hi), sizes.astype("int64"))
