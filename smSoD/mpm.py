import heapq
import math

import numpy as np

DENSE_LIMIT = 1 << 22
U64_MAX = float(2 ** 64 - 1)


class MPMBuilder:
    def __init__(self, N, gpu=None, search="greedy", max_group=0):
        self.N = N
        self.gpu = gpu
        self.search = search
        self.max_group = max_group
        self.log_n1 = math.log2(N + 1.0)
        self.nlogn = N * math.log2(N)

    def model_cost(self, table_size):
        return self.log_n1 * (table_size - 1.0)

    def pop_cost(self, keys):
        # N * entropy of each row
        G, N = keys.shape
        s = np.sort(keys, axis=1)
        new = np.ones((G, N), dtype=bool)
        new[:, 1:] = s[:, 1:] != s[:, :-1]
        first = np.nonzero(new.ravel())[0]
        cnt = np.diff(np.append(first, G * N)).astype(np.float64)
        clogc = np.bincount(first // N, weights=cnt * np.log2(cnt), minlength=G)
        return self.nlogn - clogc

    def pair_cost_cpu(self, codes, card, a, b):
        N = self.N
        out = np.empty(len(a))
        step = max(1, DENSE_LIMIT // N)
        for s in range(0, len(a), step):
            ia = a[s:s + step]
            ib = b[s:s + step]
            cb = card[ib]
            cells = card[ia] * cb
            if cells.sum() > DENSE_LIMIT:
                out[s:s + step] = self.pop_cost(codes[ia] * cb[:, None] + codes[ib])
                continue
            end = np.cumsum(cells)
            key = codes[ia] * cb[:, None] + codes[ib] + (end - cells)[:, None]
            c = np.bincount(key.ravel(), minlength=int(end[-1]))
            out[s:s + step] = self.nlogn - np.log2(c[key]).sum(axis=1)
        return out

    def pair_cost(self, codes, card, a, b):
        if self.gpu is not None and len(a) * self.N >= self.gpu.gpu_min:
            return self.nlogn - self.gpu.pair_clogc(codes, card, a, b, self.N)
        return self.pair_cost_cpu(codes, card, a, b)

    def build(self, book):
        N = self.N
        m = book.local.shape[0]
        cap = 2 * m - 1

        codes = np.empty((cap, N), dtype=np.int64)
        codes[:m] = book.local
        card = np.zeros(cap, dtype=np.int64)
        card[:m] = book.sizes
        table = np.zeros(cap)
        table[:m] = book.sizes
        size = np.zeros(cap, dtype=np.int64)
        size[:m] = 1
        mdl = np.zeros(cap)
        mdl[:m] = self.model_cost(table[:m]) + self.pop_cost(book.local)
        alive = np.zeros(cap, dtype=bool)
        alive[:m] = True
        members = [[i] for i in range(m)]

        def gain_of(a, b):
            if self.max_group > 0:
                ok = size[a] + size[b] <= self.max_group
                a, b = a[ok], b[ok]
            if len(a) == 0:
                return a, b, np.empty(0)
            t = np.minimum(table[a] * table[b], U64_MAX)
            pc = self.pair_cost(codes, card, a, b)
            return a, b, mdl[a] + mdl[b] - (self.model_cost(t) + pc)

        def merge(i, j, n):
            u, inv = np.unique(codes[i] * card[j] + codes[j], return_inverse=True)
            codes[n] = inv.ravel()
            card[n] = u.size
            table[n] = min(table[i] * table[j], U64_MAX)
            size[n] = size[i] + size[j]
            mdl[n] = self.model_cost(table[n]) + self.pop_cost(codes[n:n + 1])[0]
            members.append(sorted(members[i] + members[j]))
            alive[i] = False
            alive[j] = False
            alive[n] = True

        if m > 1:
            a, b = np.triu_indices(m, 1)
            a, b, gain = gain_of(a, b)
        else:
            a = b = gain = np.empty(0)

        if self.search == "gm":
            self.gm_search(a, b, gain, cap, m, alive, gain_of, merge)
        else:
            self.greedy_search(a, b, gain, m, alive, gain_of, merge)

        return [members[i] for i in np.nonzero(alive)[0]]

    def greedy_search(self, a, b, gain, m, alive, gain_of, merge):
        # merge the pair with the largest gain, one pair per step
        heap = [(-g, i, j) for g, i, j in zip(gain.tolist(), a.tolist(), b.tolist()) if g > 0]
        heapq.heapify(heap)
        n = m
        while heap:
            _, i, j = heapq.heappop(heap)
            if not alive[i] or not alive[j]:
                continue
            merge(i, j, n)
            others = np.nonzero(alive[:n])[0]
            if others.size > 0:
                oa, ob, g = gain_of(others, np.full(others.size, n))
                for t, gn in zip(oa.tolist(), g.tolist()):
                    if gn > 0:
                        heapq.heappush(heap, (-gn, t, n))
            n += 1

    def gm_search(self, a, b, gain, cap, m, alive, gain_of, merge):
        # greedy mating: every group picks its best partner,
        # pairs that pick each other are merged in the same step
        G = np.full((cap, cap), -np.inf)
        G[a, b] = gain
        G[b, a] = gain
        n = m
        while True:
            idx = np.nonzero(alive[:n])[0]
            if idx.size < 2:
                break
            sub = G[np.ix_(idx, idx)]
            best = sub.argmax(axis=1)
            x = np.arange(idx.size)
            mutual = (sub[x, best] > 0) & (best[best] == x) & (x < best)
            pairs = list(zip(idx[x[mutual]].tolist(), idx[best[mutual]].tolist()))
            if not pairs:
                break

            new = []
            for i, j in pairs:
                merge(i, j, n)
                G[[i, j], :] = -np.inf
                G[:, [i, j]] = -np.inf
                new.append(n)
                n += 1

            new = np.array(new)
            now = np.nonzero(alive[:n])[0]
            A = np.repeat(new, now.size)
            B = np.tile(now, new.size)
            keep = (A != B) & ~(np.isin(B, new) & (B < A))
            A, B, g = gain_of(A[keep], B[keep])
            G[A, B] = g
            G[B, A] = g
