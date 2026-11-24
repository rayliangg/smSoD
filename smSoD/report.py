import csv
import os
import platform
import sys

import numpy as np

FIELDS = ["backend", "m", "k", "run", "seed", "pop_size", "max_gen", "error", "nfe", "seconds",
          "gpu_calls"]
START = "<!-- PERFORMANCE:START -->"
END = "<!-- PERFORMANCE:END -->"


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def machine_info(backends):
    info = ["Python %d.%d" % sys.version_info[:2], "NumPy " + np.__version__]

    cpu = platform.machine()
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":")[1].strip()
                break
    info.append("CPU: %s (%d cores)" % (cpu, os.cpu_count()))

    if "cupy" in backends:
        import cupy as cp
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode()
        info.append("CuPy " + cp.__version__)
        info.append("GPU: " + name)
    return info


def mean_std(x):
    x = np.array(x, dtype=float)
    return x.mean(), x.std()


def make_table(rows, pop_size, max_gen, info, workers=1):
    rows = [r for r in rows if int(r["pop_size"]) == pop_size and int(r["max_gen"]) == max_gen]
    backends = [b for b in ["numpy", "cupy"] if any(r["backend"] == b for r in rows)]
    title = {"numpy": "NumPy", "cupy": "CuPy"}

    res = {}
    for r in rows:
        res.setdefault((int(r["m"]), int(r["k"]), r["backend"]), []).append(r)

    head = ["m", "k", "len"]
    for b in backends:
        head += [title[b] + " error (mean ± std)", title[b] + " time / run (s)"]
    if len(backends) == 2:
        head.append("Speedup")
    lines = ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)]

    for m, k in sorted(set((m, k) for m, k, b in res)):
        line = [str(m), str(k), str(m * k)]
        t = {}
        for b in backends:
            rs = res.get((m, k, b))
            if not rs:
                line += ["-", "-"]
                continue
            mu, sd = mean_std([float(r["error"]) for r in rs])
            t[b] = mean_std([float(r["seconds"]) for r in rs])[0]
            line += ["%.3e ± %.1e (n=%d)" % (mu, sd, len(rs)), "%.2f" % t[b]]
        if len(backends) == 2:
            line.append("%.2fx" % (t["numpy"] / t["cupy"]) if len(t) == 2 else "-")
        lines.append("| " + " | ".join(line) + " |")

    text = ("`pop_size=%d`, `max_gen=%d` (NFE = %d per run). Error is the best fitness "
            "at the end of a run (optimum = 0). Speedup = NumPy time / CuPy time."
            % (pop_size, max_gen, pop_size * max_gen))
    if workers > 1:
        text += " %d runs were run at the same time, the time per run includes sharing the CPU and GPU." % workers
    else:
        text += " Runs were run one at a time."

    out = [text, ""]
    if info:
        out += ["Environment: " + "; ".join(info), ""]
    return "\n".join(out + lines)


def update_readme(path, table):
    text = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    block = START + "\n" + table + "\n" + END
    if START in text and END in text:
        before = text.split(START)[0]
        after = text.split(END)[1]
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n## Performance\n\n" + block + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
