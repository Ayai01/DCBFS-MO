# DCBFS-MO

Compiled distribution of the DCBFS-MO algorithm for constrained multiobjective optimization.

## Status

The algorithm is currently distributed **in compiled form only**.
The complete source code will be released together with the paper upon acceptance.
Benchmark definitions and data are open.

## Quick start (Windows x64)

```bat
run.cmd list
run.cmd --list-methods
run.cmd run --case ZDT1 --method final --seed 1700000001 --budget 600 --backend python
```

No Python installation is required.

## Contents

| Path | Content |
| --- | --- |
| `DCBFS-MO.exe`, `_internal\\` | Standalone runtime with the compiled algorithm |
| `src\\dcbfs\\problems\\` | Benchmark definitions and evaluators |
| `data\\` | Task catalog, reference fronts and normalization assets |
| `run.cmd` | Launcher with deterministic thread settings |

## Notes

- This build targets Windows x64.
- `--backend auto` and `--backend cpp` use the qualified FP32 native profile.
- `--backend python` retains reference Python numerical behavior through native compiled modules.
- Full algorithm source will be published with the paper upon acceptance.
