import argparse
import os
import sys
from dataclasses import fields

from .backend import get_backend
from .config import Config
from .optimizer import run


def make_parser(backend="auto"):
    d = Config()
    p = argparse.ArgumentParser(description="smSoD (ECGA + split-on-demand)")
    p.add_argument("--m", type=int, default=d.m, help="number of subproblems")
    p.add_argument("--k", type=int, default=d.k, help="dim of each subproblem (len = m*k)")
    p.add_argument("--pop_size", type=int, default=d.pop_size)
    p.add_argument("--tour_size", type=int, default=d.tour_size)
    p.add_argument("--cross_prob", type=float, default=d.cross_prob)
    p.add_argument("--max_gen", type=int, default=d.max_gen)
    p.add_argument("--split_rate", type=float, default=d.split_rate)
    p.add_argument("--decrease_rate", type=float, default=d.decrease_rate)
    p.add_argument("--min_bound", type=float, default=d.min_bound)
    p.add_argument("--lower", type=float, default=d.lower)
    p.add_argument("--upper", type=float, default=d.upper)
    p.add_argument("--seed", type=int, default=None, help="random if not given")
    p.add_argument("--group_crossover", action="store_true",
                   help="experimental: one donor parent per MPM group")
    p.add_argument("--backend", choices=["auto", "numpy", "cupy"], default=backend)
    p.add_argument("--output_file", default="mk.csv", help="csv to append the result ('' = no csv)")
    p.add_argument("--verbose", type=int, default=0, metavar="N",
                   help="print every N generations")
    return p


def save_csv(path, cfg, res):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    first = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a") as f:
        if first:
            f.write("m,k,length,mode,eva_fun,seed,error,backend,seconds\n")
        f.write("%d,%d,%d,smSoD,%d,%d,%.16e,%s,%.4f\n" % (
            cfg.m, cfg.k, cfg.length, res.nfe, res.seed, res.error, res.backend, res.seconds))


def main(argv=None, backend="auto"):
    args = make_parser(backend).parse_args(argv)
    names = [f.name for f in fields(Config)]
    cfg = Config(**{n: getattr(args, n) for n in names})

    try:
        cfg.check()
        name, _ = get_backend(args.backend)
    except (ValueError, RuntimeError) as e:
        sys.exit("error: %s" % e)

    print("smSoD  backend=%s  m=%d  k=%d  pop=%d  gens=%d" % (
        name, cfg.m, cfg.k, cfg.pop_size, cfg.max_gen), flush=True)
    res = run(cfg, name, args.verbose)
    print("error=%.6e  nfe=%d  time=%.2fs  seed=%d" % (res.error, res.nfe, res.seconds, res.seed))

    if args.output_file:
        save_csv(args.output_file, cfg, res)
    return res
