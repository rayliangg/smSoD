import argparse
import os
import time

from smSoD import Config, get_backend, run
from smSoD import report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--k", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--backends", nargs="+", default=["numpy", "cupy"])
    p.add_argument("--pop_size", type=int, default=200)
    p.add_argument("--max_gen", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mpm_search", choices=["greedy", "gm"], default="greedy", help="cupy runs only")
    p.add_argument("--max_group", type=int, default=0, help="cupy runs only")
    p.add_argument("--csv", default="results/benchmark.csv")
    p.add_argument("--readme", default="README.md")
    p.add_argument("--no_readme", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--report_only", action="store_true")
    args = p.parse_args()

    backends = []
    for b in args.backends:
        try:
            backends.append(get_backend(b)[0])
        except RuntimeError as e:
            print("skip", b, ":", e)
    backends = sorted(set(backends), key=backends.index)

    if not args.report_only:
        if args.fresh and os.path.exists(args.csv):
            os.remove(args.csv)

        # runs already in the csv are skipped
        done = set()
        for r in report.load_rows(args.csv):
            done.add((r["backend"], r["m"], r["k"], r["run"], r["pop_size"], r["max_gen"]))

        jobs = []
        for m in args.m:
            for k in args.k:
                for r in range(args.runs):
                    for b in backends:
                        if (b, str(m), str(k), str(r), str(args.pop_size), str(args.max_gen)) not in done:
                            jobs.append((b, m, k, r))
        print("%d runs, %d done before" % (len(jobs), len(done)))

        # first cupy call compiles the kernels, don't time it
        if "cupy" in backends:
            run(Config(m=4, k=2, max_gen=5, seed=0, gpu_min=0), "cupy")

        t0 = time.time()
        for i, (b, m, k, r) in enumerate(jobs):
            cfg = Config(m=m, k=k, pop_size=args.pop_size, max_gen=args.max_gen, seed=args.seed + r)
            if b == "cupy":
                cfg.mpm_search = args.mpm_search
                cfg.max_group = args.max_group
            res = run(cfg, b)
            report.append_row(args.csv, {
                "backend": b, "m": m, "k": k, "run": r, "seed": cfg.seed,
                "pop_size": args.pop_size, "max_gen": args.max_gen,
                "error": "%.16e" % res.error, "nfe": res.nfe, "seconds": "%.4f" % res.seconds,
                "gpu_calls": res.gpu_calls,
            })
            left = (time.time() - t0) / (i + 1) * (len(jobs) - i - 1)
            print("[%d/%d] %s m=%d k=%d run=%d error=%.3e %.1fs gpu_calls=%d (%.0f min left)" % (
                i + 1, len(jobs), b, m, k, r, res.error, res.seconds, res.gpu_calls, left / 60), flush=True)
        print("total %.1f min" % ((time.time() - t0) / 60))

    rows = report.load_rows(args.csv)
    if not rows:
        print("no results in", args.csv)
        return
    used = set(r["backend"] for r in rows) & set(backends)
    table = report.make_table(rows, args.pop_size, args.max_gen, report.machine_info(used))
    print(table)
    if not args.no_readme:
        report.update_readme(args.readme, table)
        print("README updated")


if __name__ == "__main__":
    main()
