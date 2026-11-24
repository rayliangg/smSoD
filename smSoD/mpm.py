import heapq
import math

import numpy as np

DENSE_LIMIT = 1 << 22
U64_MAX = float(2 ** 64 - 1)


class MPMBuilder:
    def __init__(self, N, gpu=None, gpu_min=1 << 17):
        self.N = N
        self.gpu = gpu
        self.gpu_min = gpu_min
        self.gpu_calls = 0
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

    def pair_pop_cost(self, codes, card, a, b):
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
            if self.gpu is not None and len(ia) * N >= self.gpu_min:
                out[s:s + step] = self._pair_gpu(codes, ia, ib, cb, end - cells, int(end[-1]))
            else:
                key = codes[ia] * cb[:, None] + codes[ib] + (end - cells)[:, None]
                c = np.bincount(key.ravel(), minlength=int(end[-1]))
                out[s:s + step] = self.nlogn - np.log2(c[key]).sum(axis=1)
        return out

    def _pair_gpu(self, codes, ia, ib, cb, offset, n_cell):
        # only send the rows we need, the pairs are built on the gpu
        cp = self.gpu
        self.gpu_calls += 1
        rows, inv = np.unique(np.concatenate([ia, ib]), return_inverse=True)
        inv = inv.ravel()
        dc = cp.asarray(codes[rows])
        da = cp.asarray(inv[:len(ia)])
        db = cp.asarray(inv[len(ia):])
        key = dc[da] * cp.asarray(cb)[:, None] + dc[db] + cp.asarray(offset)[:, None]
        c = cp.bincount(key.ravel(), minlength=n_cell)
        clogc = cp.log2(c[key].astype(cp.float64)).sum(axis=1)
        return self.nlogn - clogc.get()

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

            u, inv = np.unique(codes[i] * card[j] + codes[j], return_inverse=True)
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
