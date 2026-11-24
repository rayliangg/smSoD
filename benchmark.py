import argparse
import multiprocessing as mp
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from smSoD import Config, get_backend, run
from smSoD import report

warmed = set()


def n_gpus():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def run_one(job):
    backend, m, k, r, gpu_id, args = job
    if backend == "cupy":
        import cupy
        cupy.cuda.Device(gpu_id).use()
    if (backend, gpu_id) not in warmed:
        run(Config(m=2, k=2, pop_size=args.pop_size, max_gen=5, seed=0, gpu_min=0), backend)
        warmed.add((backend, gpu_id))
    cfg = Config(m=m, k=k, pop_size=args.pop_size, max_gen=args.max_gen, seed=args.seed + r)
    return job, run(cfg, backend)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, nargs="+", default=[100, 200, 300])
    p.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--backends", nargs="+", default=["numpy", "cupy"])
    p.add_argument("--pop_size", type=int, default=200)
    p.add_argument("--max_gen", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--gpus", type=int, default=None)
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

    gpus = n_gpus() if "cupy" in backends else 0
    if args.gpus is not None:
        gpus = min(gpus, args.gpus)

    if not args.report_only:
        if args.fresh and os.path.exists(args.csv):
            os.remove(args.csv)

        # runs already in the csv are skipped
        done = set()
        for r in report.load_rows(args.csv):
            done.add((r["backend"], r["m"], r["k"], r["run"], r["pop_size"], r["max_gen"]))

        jobs = []
        i_gpu = 0
        for m in sorted(args.m, reverse=True):
            for k in sorted(args.k, reverse=True):
                for b in backends:
                    for r in range(args.runs):
                        key = (b, str(m), str(k), str(r), str(args.pop_size), str(args.max_gen))
                        if key in done:
                            continue
                        gpu_id = 0
                        if b == "cupy":
                            gpu_id = i_gpu % max(gpus, 1)
                            i_gpu += 1
                        jobs.append((b, m, k, r, gpu_id, args))

        print("%d runs, %d done before, %d workers, %d gpus" % (len(jobs), len(done), args.workers, gpus))

        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for i, (job, res) in enumerate(pool.imap_unordered(run_one, jobs)):
                b, m, k, r, gpu_id = job[:5]
                report.append_row(args.csv, {
                    "backend": b, "m": m, "k": k, "run": r, "seed": args.seed + r,
                    "pop_size": args.pop_size, "max_gen": args.max_gen,
                    "error": "%.16e" % res.error, "nfe": res.nfe, "seconds": "%.4f" % res.seconds,
                    "gpu_calls": res.gpu_calls, "workers": args.workers,
                })
                left = (time.time() - t0) / (i + 1) * (len(jobs) - i - 1)
                dev = "gpu%d" % gpu_id if b == "cupy" else "cpu"
                print("[%d/%d] %s %s m=%d k=%d run=%d error=%.3e %.1fs (%.0f min left)" % (
                    i + 1, len(jobs), b, dev, m, k, r, res.error, res.seconds, left / 60), flush=True)
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
