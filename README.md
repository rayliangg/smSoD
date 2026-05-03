# smSoD (C++)

C++17 implementation of **smSoD**

**Please cite:** *Novel Discretization Scheme for Multidimensional Split-on-Demand on Real-Valued Optimization with High Multi-modality.*

```bash
make && ./main --m 5 --k 2
```

`--m` / `--k` set subproblem count and block size (`len = m*k`). More options live in `parameter.*`. Results append to `../mk.csv` by default.

## License

GNU GPL v3.0 (`GPL-3.0-only`). See `LICENSE`.
