import argparse
import os
import time

from smSoD import Config, get_backend, run
from smSoD import report


def usable_backends(wanted):
    ok = []
    for b in wanted:
        try:
            name, _ = get_backend(b)
            if name not in ok:
                ok.append(name)
        except RuntimeError as e:
            print("[skip] %s: %s" % (b, e))
    return ok


def main():
    p = argparse.ArgumentParser(description="smSoD benchmark (NumPy vs CuPy)")
    p.add_argument("--m", type=int, nargs="+", default=[1, 2, 5, 50, 100])
    p.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--backends", nargs="+", choices=["numpy", "cupy"], default=["numpy", "cupy"])
    p.add_argument("--pop_size", type=int, default=200)
    p.add_argument("--max_gen", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="run r uses seed + r")
    p.add_argument("--csv", default="results/benchmark.csv")
    p.add_argument("--readme", default="README.md")
    p.add_argument("--no_readme", action="store_true")
    p.add_argument("--fresh", action="store_true", help="delete old csv and start again")
    p.add_argument("--report_only", action="store_true")
    args = p.parse_args()

    backends = usable_backends(args.backends)

    if not args.report_only:
        if not backends:
            raise SystemExit("no backend can be used")
        if args.fresh and os.path.exists(args.csv):
            os.remove(args.csv)

        done = set()
        for r in report.load_rows(args.csv):
            done.add((r["backend"], int(r["m"]), int(r["k"]), int(r["run"]),
                      int(r["pop_size"]), int(r["max_gen"])))
        todo = []
        for m in args.m:
            for k in args.k:
                for b in backends:
                    for r in range(args.runs):
                        if (b, m, k, r, args.pop_size, args.max_gen) not in done:
                            todo.append((b, m, k, r))
        print("%d runs to do (%d already in %s)" % (len(todo), len(done), args.csv), flush=True)

        warm = set()
        t0 = time.perf_counter()
        for i, (b, m, k, r) in enumerate(todo, 1):
            if b not in warm:
                run(Config(m=2, k=2, pop_size=args.pop_size, max_gen=5, seed=0), b)
                warm.add(b)

            cfg = Config(m=m, k=k, pop_size=args.pop_size, max_gen=args.max_gen, seed=args.seed + r)
            res = run(cfg, b)
            report.append_row(args.csv, {
                "backend": b, "m": m, "k": k, "run": r, "seed": cfg.seed,
                "pop_size": args.pop_size, "max_gen": args.max_gen,
                "error": "%.16e" % res.error, "nfe": res.nfe, "seconds": "%.4f" % res.seconds,
            })

            eta = (time.perf_counter() - t0) / i * (len(todo) - i)
            print("[%4d/%d] %-5s m=%-3d k=%d run %d  error=%.3e  %7.2fs  ETA %.1f min" % (
                i, len(todo), b, m, k, r, res.error, res.seconds, eta / 60), flush=True)

    rows = report.load_rows(args.csv)
    if not rows:
        raise SystemExit("no results in " + args.csv)

    have = set(r["backend"] for r in rows) & set(backends)
    table = report.make_table(rows, args.pop_size, args.max_gen, report.machine_info(have))
    print()
    print(table)
    if not args.no_readme:
        report.update_readme(args.readme, table)
        print("\nupdated " + args.readme)


if __name__ == "__main__":
    main()
