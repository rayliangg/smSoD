# smSoD

Python code of smSoD (ECGA with multidimensional split-on-demand discretization).
There are two versions: NumPy (CPU) and CuPy (GPU).

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/rayliangg/smSoD/blob/main/smSoD_kaggle.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rayliangg/smSoD/blob/main/smSoD_colab.ipynb)

## Citation

Che-Wei Liang and Tian-Li Yu, "Novel Discretization Scheme for Multidimensional
Split-on-Demand on Real-Valued Optimization with High Multi-modality," in
Proc. International Joint Conference on Computational Intelligence, Springer,
2025, pp. 281-294.

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

The GPU model building in the CuPy version follows:

```bibtex
@inproceedings{shao2013speeding,
  title={Speeding up model building for ECGA on CUDA platform},
  author={Shao, Chung-Yu and Yu, Tian-Li},
  booktitle={Proceedings of the 15th Annual Conference on Genetic and Evolutionary Computation},
  pages={1197--1204},
  year={2013}
}
```

## Install

```bash
pip install numpy==2.3.3
pip install cupy-cuda12x==13.6.0 fastrlock==0.8.3   # only for the GPU version
```

## Run

```bash
python smSoD_numpy.py --m 5 --k 2
python smSoD_cupy.py --m 5 --k 2
```

`--m` is the number of subproblems, `--k` the size of a subproblem (length = m*k).
The result is printed and added to `mk.csv`. Other options:

- `--pop_size` (200), `--max_gen` (2000), `--tour_size` (8), `--cross_prob` (0.975)
- `--split_rate` (0.5), `--decrease_rate` (0.998), `--min_bound` (1e-5)
- `--lower` / `--upper` (-200 / 200), `--seed` (random)
- `--verbose N`: print every N generations
- `--output_file`: csv file, `''` = do not save
- `--group_crossover`: one parent per MPM group (experimental)

CuPy version only:

- `--gpu_min` (65536): a step runs on the GPU only if it has at least this many elements
- `--mpm_search greedy|gm`: greedy search of ECGA, or greedy mating (GM) search from Shao & Yu 2013
- `--max_group` (0): max subproblems in one MPM group (MAXL in Shao & Yu 2013), 0 = no limit

`main.py` uses the GPU version if CuPy works, otherwise the NumPy version.

## Kaggle

1. Open `smSoD_kaggle.ipynb` on Kaggle (badge above).
2. Settings: Accelerator = GPU P100 or GPU T4, Internet = On.
3. Change the settings cell if needed, then Save Version -> Save & Run All.
4. `benchmark.csv` and `README.md` are in the Output tab.

Kaggle stops a run after 12 hours. If the benchmark is too long, run `numpy` and `cupy` in
two versions, put both csv files together and run `python benchmark.py --report_only`.

## Colab

1. Open `smSoD_colab.ipynb` on Colab (badge above).
2. Runtime -> Change runtime type -> T4 GPU.
3. Run the cells in order. For the full benchmark set `USE_DRIVE = True` in the settings cell,
   then the csv is saved to Google Drive after every run and the benchmark cell can be run
   again after a disconnect.
4. The last cell downloads `benchmark.csv` and `README.md`.

## Benchmark

```bash
python benchmark.py --m 50 100 200 --k 2 4 8 --runs 5
```

`os.cpu_count()` runs are done at the same time (`--workers` to change it), the CuPy runs
share the one GPU. Every run is saved to `results/benchmark.csv` when it finishes, so it can
be stopped and started again. At the end the table below is updated.
`--report_only` only rebuilds the table, `--fresh` starts from an empty csv.

## Code

```
smSoD/
  config.py       parameters
  backend.py      numpy / cupy
  problem.py      test function
  selection.py    tournament selection
  discretize.py   split-on-demand
  mpm.py          marginal product model
  crossover.py    make offspring
  gpu.py          the parts that run on the GPU (CuPy version)
  optimizer.py    main loop
  cli.py          command line
  report.py       benchmark table
```

The GPU is only used for the two slowest steps: scoring all pairs of subproblems when
building the MPM (all pairs at the same time like gECGA), and scoring the split points when
a tree level has many points. Everything else is faster on the CPU. Only CuPy array
functions are used, and one GPU. FastCount from Shao & Yu 2013 is not used because our
symbols are leaf ids, not bits. NumPy and CuPy give the same result for the same seed.

## Performance

<!-- PERFORMANCE:START -->
Not run yet.
<!-- PERFORMANCE:END -->

## License

GPL-3.0, see `LICENSE`.
