# smSoD

Python implementation of **smSoD**: the extended compact genetic algorithm (ECGA)
with multidimensional split-on-demand discretization for real-valued optimization.
The same code runs on **NumPy (CPU)** and **CuPy (GPU)**.

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/rayliangg/smSoD/blob/main/smSoD_kaggle.ipynb)

## Citation

If you use this code, please cite:

Che-Wei Liang and Tian-Li Yu, “Novel Discretization Scheme for Multidimensional
Split-on-Demand on Real-Valued Optimization with High Multi-modality,” in
*Proc. International Joint Conference on Computational Intelligence*, Springer,
2025, pp. 281–294.

```bibtex
@inproceedings{liang2025novel,
  title={Novel Discretization Scheme for Multidimensional Split-on-Demand on Real-Valued Optimization with High Multi-modality},
  author={Liang, Che-Wei and Yu, Tian-Li},
  booktitle={International Joint Conference on Computational Intelligence},
  pages={281--294},
  year={2025},
  organization={Springer}
}
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Run smSoD, uses CuPy if a GPU is available, otherwise NumPy |
| `smSoD_numpy.py` | NumPy (CPU) version |
| `smSoD_cupy.py` | CuPy (GPU) version |
| `benchmark.py` | Runs the m × k grid on both backends and writes the Performance table below |
| `smSoD_kaggle.ipynb` | Ready-to-run Kaggle notebook (free GPU) |
| `requirements.txt` | Library versions |
| `smSoD/` | The package (see below) |

```
smSoD/
├── config.py       parameters of a run
├── backend.py      NumPy / CuPy switch and random numbers
├── problem.py      test function (separable chained ridge)
├── selection.py    tournament selection
├── discretize.py   multidimensional split-on-demand -> codebook
├── mpm.py          marginal product model (greedy MDL merge)
├── crossover.py    sample offspring from the codebook
├── optimizer.py    main loop, run()
├── cli.py          command line options
└── report.py       benchmark table -> README
```

## How to run

### Kaggle (free GPU)

1. Click the **Open in Kaggle** badge above (or on Kaggle: **Create → New Notebook →
   File → Import Notebook**, and paste the GitHub link of `smSoD_kaggle.ipynb`).
2. In **Settings**, set **Accelerator** to *GPU T4 x2* or *GPU P100* and turn **Internet** on
   (Internet needs a phone-verified account).
3. Run the cells from top to bottom. The notebook replaces Kaggle's CuPy with CuPy 13.6.0
   (CUDA 12) and installs the CUDA 12 NVRTC / cuBLAS / cuRAND libraries from pip, since the
   Kaggle image only ships CUDA 13.
4. For the full benchmark use **Save Version → Save & Run All (Commit)** so it runs in the
   background. The updated `README.md` and `benchmark.csv` appear in the **Output** tab.

Kaggle gives about 30 GPU hours per week, and one session runs at most 12 hours.

### Locally

```bash
pip install -r requirements.txt     # numpy==2.3.3
pip install cupy-cuda12x==13.6.0 fastrlock==0.8.3    # GPU only, needs the CUDA 12 toolkit (NVRTC)
```

```bash
python smSoD_numpy.py --m 5 --k 2
python smSoD_cupy.py  --m 5 --k 2
python main.py        --m 50 --k 4 --verbose 100   # CuPy if a GPU is usable, else NumPy
```

`--m` is the number of subproblems and `--k` the block size (problem length = `m*k`).
Each run prints the final error and appends a row to `mk.csv`
(`m,k,length,mode,eva_fun,seed,error,backend,seconds`).

| Option | Default | Meaning |
|---|---|---|
| `--m`, `--k` | 5, 2 | subproblem count and size (`1 ≤ k ≤ 62`) |
| `--pop_size` | 200 | population size |
| `--max_gen` | 2000 | generations (NFE = `pop_size × max_gen`) |
| `--tour_size` | 8 | tournament size |
| `--cross_prob` | 0.975 | fraction of offspring; the rest are elites |
| `--split_rate`, `--decrease_rate` | 0.5, 0.998 | stop splitting below `pop_size × split_rate` points; decay per generation |
| `--min_bound` | 1e-5 | minimum region width (adapted during the run) |
| `--lower`, `--upper` | -200, 200 | search range |
| `--seed` | random | RNG seed |
| `--gpu_min` | 131072 | CuPy: a step goes to the GPU only if it has at least this many elements |
| `--output_file` | `mk.csv` | result CSV (`''` disables it) |
| `--verbose N` | 0 | print progress every N generations |
| `--group_crossover` | off | experimental: one donor parent per MPM group instead of per subproblem |

As a library:

```python
from smSoD import Config, run

res = run(Config(m=5, k=2, seed=1), backend="cupy")   # or "numpy" / "auto"
print(res.error, res.nfe, res.seconds)
```

### Benchmark

```bash
python benchmark.py                                   # m ∈ {50,100,200}, k ∈ {2,3,4,5}, 5 runs, both backends
python benchmark.py --backends cupy --runs 3 --max_gen 500
python benchmark.py --report_only                     # rebuild the table from results/benchmark.csv
python benchmark.py --workers 4 --gpus 2               # 4 runs at a time, CuPy runs spread over 2 GPUs
```

Runs are independent, so the benchmark runs `--workers` of them at the same time (default:
one per CPU core) and spreads the CuPy runs over all GPUs. The results are the same as
running them one by one; only the time per run includes sharing the CPU.

Each run is appended to `results/benchmark.csv` straight away, so an interrupted
benchmark resumes when started again (`--fresh` starts over). When it finishes, the
table between the `PERFORMANCE` markers below is replaced.

## Implementation notes

* The algorithm runs on the CPU (NumPy). Every per-individual and per-point loop is a
  batched array operation; the split-on-demand tree is built level by level for all
  subproblems at once.
* The CuPy version is heterogeneous: only steps that are large enough to pay for the GPU
  launch and transfer are sent to the GPU. These are scoring all pairs of subproblems in
  the MPM (`m(m-1)/2 x pop_size` work from `m x pop_size` data) and scoring the candidate
  split points on crowded tree levels. Everything else stays on the CPU, so small problems
  run at NumPy speed and the GPU helps when `m`, `k` or `pop_size` grow. `--gpu_min` sets
  the size threshold, and the run prints how many GPU calls it made.
* NumPy and CuPy give the same result for the same seed.

## Performance

<!-- PERFORMANCE:START -->
Not generated yet. Run `python benchmark.py` (or the Kaggle notebook) to fill in this table.
<!-- PERFORMANCE:END -->

## License

GNU GPL v3.0 (`GPL-3.0-only`). See `LICENSE`.
