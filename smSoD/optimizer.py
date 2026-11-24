import copy
import os
import time

import numpy as np

from .backend import get_backend
from .crossover import sample_offspring
from .discretize import Discretizer
from .gpu import GPU
from .mpm import MPMBuilder
from .problem import RidgeProblem
from .selection import tournament


class Result:
    def __init__(self, error, nfe, seconds, backend, seed, best, gpu_calls=0):
        self.error = error
        self.nfe = nfe
        self.seconds = seconds
        self.backend = backend
        self.seed = seed
        self.best = best
        self.gpu_calls = gpu_calls


class smSoD:
    # cp = cupy module for the cupy version, None for numpy
    def __init__(self, cfg, cp=None):
        cfg.check()
        if cp is None and (cfg.mpm_search != "greedy" or cfg.max_group > 0):
            raise ValueError("mpm_search=gm and max_group are only in the CuPy version")
        self.cfg = cfg
        self.gpu = None
        if cp is not None:
            self.gpu = GPU(cp, cfg.gpu_min)
        self.rng = np.random.default_rng(cfg.seed)

        self.split_rate = cfg.split_rate
        self.min_bound = cfg.min_bound
        self.nfe = 0

        N, L = cfg.pop_size, cfg.length
        self.genes = cfg.lower + (cfg.upper - cfg.lower) * self.rng.random((N, L))
        self.fitness = np.full(N, np.inf)

        self.problem = RidgeProblem(cfg, self.rng)
        self.discretize = Discretizer(cfg, self.rng, self.gpu)
        self.mpm = MPMBuilder(cfg.pop_size, self.gpu, cfg.mpm_search, cfg.max_group)

    @property
    def gpu_calls(self):
        if self.gpu is None:
            return 0
        return self.gpu.calls

    def evaluate(self):
        self.fitness = self.problem(self.genes)
        self.nfe += self.cfg.pop_size

    def adjust_min_bound(self):
        q = self.cfg.pop_size // 4
        if q == 0:
            return
        top = np.partition(self.fitness, q - 1)[:q]
        if top.std() < self.min_bound:
            self.min_bound = max(self.min_bound / 1.05, 1e-13)

    def step(self):
        cfg = self.cfg
        N, m, k = cfg.pop_size, cfg.m, cfg.k

        self.evaluate()
        self.adjust_min_bound()

        win = tournament(self.fitness, cfg.tour_size, self.rng)
        pool = self.genes[win]
        pool_fit = self.fitness[win]

        points = pool.reshape(N, m, k).transpose(1, 0, 2)
        book = self.discretize(points, self.split_rate, self.min_bound)
        self.split_rate *= cfg.decrease_rate
        self.n_leaf = int(book.sizes.sum())

        groups = self.mpm.build(book)

        n_child = int(N * cfg.cross_prob)
        n_elite = N - n_child
        new = np.empty_like(self.genes)
        new[:n_child] = sample_offspring(book, n_child, groups, self.rng, cfg.group_crossover)
        if n_elite > 0:
            elite = np.argpartition(pool_fit, n_elite - 1)[:n_elite]
            new[n_child:] = pool[elite]
        self.genes = new
        return groups

    def run(self, verbose=0):
        t0 = time.perf_counter()
        for gen in range(1, self.cfg.max_gen):
            groups = self.step()
            if verbose and gen % verbose == 0:
                print("gen %5d  best %.6e  groups %4d  leaves %5d  gpu calls %6d  %7.1fs" % (
                    gen, self.fitness.min(), len(groups), self.n_leaf, self.gpu_calls,
                    time.perf_counter() - t0), flush=True)
        self.evaluate()
        b = int(np.argmin(self.fitness))
        return float(self.fitness[b]), self.genes[b].copy()


def run(cfg, backend="auto", verbose=0):
    name, cp = get_backend(backend)
    cfg = copy.copy(cfg)
    if cfg.seed is None:
        cfg.seed = int.from_bytes(os.urandom(4), "little")

    t0 = time.perf_counter()
    opt = smSoD(cfg, cp)
    err, best = opt.run(verbose)
    return Result(err, opt.nfe, time.perf_counter() - t0, name, cfg.seed, best, opt.gpu_calls)
