# smSoD (C++)

C++ implementation of **smSoD**

**Please cite**

Che-Wei Liang and Tian-Li Yu, “Novel Discretization Scheme for Multidimensional Split-on-Demand on Real-Valued Optimization with High Multi-modality,” in *Proc. International Joint Conference on Computational Intelligence*, Springer, 2025, pp. 281–294.

BibTeX:

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

```bash
make && ./main --m 5 --k 2
```

`--m` / `--k` set subproblem count and block size (`len = m*k`). More options live in `parameter.*`. Results append to `../mk.csv` by default.

## License

GNU GPL v3.0 (`GPL-3.0-only`). See `LICENSE`.
