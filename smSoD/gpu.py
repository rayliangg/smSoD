import numpy as np

N_TRIAL = 5
MAX_CELLS = 1 << 25   # chunk size limits
MAX_KEYS = 1 << 24


# gpu part of the cupy version
class GPU:
    def __init__(self, cp, gpu_min):
        self.cp = cp
        self.gpu_min = gpu_min
        self.calls = 0
        self.pow2 = {}

    def trial_score(self, X, trial, reg, G, K):
        # same as Discretizer.trial_score_cpu
        cp = self.cp
        self.calls += 1
        k = X.shape[1]
        if k not in self.pow2:
            self.pow2[k] = cp.asarray(1 << np.arange(k, dtype=np.int64))
        dX = cp.asarray(X)
        dt = cp.asarray(trial)
        dr = cp.asarray(reg)
        mask = ((dX[:, None, :] > dt[dr]) * self.pow2[k]).sum(axis=2)
        key = (dr[:, None] * N_TRIAL + cp.arange(N_TRIAL)) * K + mask
        cnt = cp.zeros(G * K, dtype=cp.int32)
        cp.add.at(cnt, key.ravel(), 1)
        return cnt.reshape(G, K).max(axis=1).get()

    def pair_clogc(self, codes, card, a, b, N):
        # sum c*log2(c) of every pair, all pairs at once like gECGA (Shao & Yu 2013)
        self.calls += 1
        cum = np.cumsum(card[a] * card[b])
        out = np.empty(len(a))
        s = 0
        while s < len(a):
            base = cum[s - 1] if s > 0 else 0
            e = int(np.searchsorted(cum, base + MAX_CELLS, side="right"))
            e = min(max(e, s + 1), s + MAX_KEYS // N, len(a))
            out[s:e] = self._pair_chunk(codes, card, a[s:e], b[s:e]).get()
            s = e
        return out

    def _pair_chunk(self, codes, card, a, b):
        cp = self.cp
        rows, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
        inv = inv.ravel()
        c = cp.asarray(codes[rows])
        cb = card[b]
        cells = card[a] * cb
        off = np.cumsum(cells) - cells
        key = c[cp.asarray(inv[:len(a)])] * cp.asarray(cb)[:, None] + c[cp.asarray(inv[len(a):])]
        key += cp.asarray(off)[:, None]
        cnt = cp.zeros(int(cells.sum()), dtype=cp.int32)
        cp.add.at(cnt, key.ravel(), 1)
        return cp.log2(cnt[key].astype(cp.float64)).sum(axis=1)
