import heapq
import math

import numpy as np

from .backend import to_numpy

DENSE_LIMIT = 1 << 22
U64_MAX = float(2 ** 64 - 1)


class MPMBuilder:
    def __init__(self, N, xp):
        self.N = N
        self.xp = xp
        self.log_n1 = math.log2(N + 1.0)
        self.nlogn = N * math.log2(N)

    def model_cost(self, table_size):
        return self.log_n1 * (table_size - 1.0)

    def pop_cost(self, keys):
        # N * entropy of each row
        xp = self.xp
        G, N = keys.shape
        s = xp.sort(keys, axis=1)
        new = xp.ones((G, N), dtype=bool)
        new[:, 1:] = s[:, 1:] != s[:, :-1]
        first = xp.nonzero(new.ravel())[0]
        cnt = xp.diff(xp.concatenate([first, xp.asarray([G * N])])).astype(xp.float64)
        clogc = xp.bincount(first // N, weights=cnt * xp.log2(cnt), minlength=G)
        return to_numpy(self.nlogn - clogc)

    def pair_pop_cost(self, codes, card, a, b):
        xp = self.xp
        N = self.N
        out = np.empty(len(a))
        step = max(1, DENSE_LIMIT // N)
        for s in range(0, len(a), step):
            ia = a[s:s + step]
            ib = b[s:s + step]
            cb = card[ib]
            cells = card[ia] * cb
            joint = codes[xp.asarray(ia)] * xp.asarray(cb)[:, None] + codes[xp.asarray(ib)]
            if cells.sum() <= DENSE_LIMIT:
                end = np.cumsum(cells)
                key = joint + xp.asarray(end - cells)[:, None]
                c = xp.bincount(key.ravel(), minlength=int(end[-1]))
                clogc = xp.log2(c[key].astype(xp.float64)).sum(axis=1)
                out[s:s + step] = to_numpy(self.nlogn - clogc)
            else:
                out[s:s + step] = self.pop_cost(joint)
        return out

    def build(self, book):
        xp = self.xp
        N = self.N
        m = book.local.shape[0]
        cap = 2 * m - 1

        codes = xp.empty((cap, N), dtype=xp.int64)
        codes[:m] = book.local
        card = np.zeros(cap, dtype=np.int64)
        card[:m] = book.sizes
        table = np.zeros(cap)
        table[:m] = book.sizes
        mdl = np.zeros(cap)
        mdl[:m] = self.model_cost(table[:m]) + self.pop_cost(book.local)
        alive = np.zeros(cap, dtype=bool)
        alive[:m] = True
        members = [[i] for i in range(m)]

        def merged_mdl(a, b):
            t = np.minimum(table[a] * table[b], U64_MAX)
            return self.model_cost(t) + self.pair_pop_cost(codes, card, a, b)

        heap = []
        if m > 1:
            a, b = np.triu_indices(m, 1)
            gain = mdl[a] + mdl[b] - merged_mdl(a, b)
            for g, i, j in zip(gain.tolist(), a.tolist(), b.tolist()):
                if g > 0:
                    heap.append((-g, i, j))
            heapq.heapify(heap)

        n = m
        while heap:
            _, i, j = heapq.heappop(heap)
            if not alive[i] or not alive[j]:
                continue

            u, inv = xp.unique(codes[i] * card[j] + codes[j], return_inverse=True)
            codes[n] = inv.ravel()
            card[n] = u.size
            table[n] = min(table[i] * table[j], U64_MAX)
            mdl[n] = self.model_cost(table[n]) + self.pop_cost(codes[n:n + 1])[0]
            members.append(sorted(members[i] + members[j]))
            alive[i] = False
            alive[j] = False

            others = np.nonzero(alive[:n])[0]
            alive[n] = True
            if others.size > 0:
                gain = mdl[others] + mdl[n] - merged_mdl(others, np.full(others.size, n))
                for t, g in zip(others.tolist(), gain.tolist()):
                    if g > 0:
                        heapq.heappush(heap, (-g, t, n))
            n += 1

        return [members[i] for i in np.nonzero(alive)[0]]
