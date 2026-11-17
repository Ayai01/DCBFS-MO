# DCBFS-MO Experimental Data for 118 Tasks and 40 Seeds

Dataset version: 2026-11-17.

This dataset accompanies *Dual-Credit Coordination and Basin-Front Memory Search for Constrained Multiobjective Optimization*.
The study contains 89 mathematical problems and 29 application tasks, 40 seeds per task and an evaluation budget of 16,000 unique designs per run.

Download and extract `dcbfs-mo-experimental-data.zip`. All measurements are UTF-8 CSV files; the package needs no algorithm runtime.

## Contents

- `runs/formal118-40seeds.csv`: 193,520 measured runs, comprising 41 configurations on 118 tasks and 40 seeds. The 27-method comparison includes DCBFS-MO (`full`) and 26 comparators; the remaining configurations support the mechanism and policy studies.
- `runs/parameter118-40seeds.csv`: 118,000 paired runs, comprising the default and 24 parameter contrasts on all 118 tasks and 40 seeds.
- `summaries/`: accepted method ranks, mechanism and policy effects, construction summaries, paired distance magnitudes, physical quantities, fixed-date portfolio results, parameter sensitivity, bidirectional dominance coverage and optimizer timing.
- `task-metadata.json`: each task's mathematical/application domain and family.
- `manifest.json`: row counts, column schemas, source digests and SHA256 digests of the exported files.

## Fields and conventions

`case`, `seed` and `configuration` or `role` identify a measurement. `original_ID` and `original_DCO_ID` retain its original provenance identifiers. `paid_FE` retains observed evaluation accounting, while `budget` is the planned cap. Missing source values remain empty CSV cells. Infinity is written as `inf`; floating-point measurements retain their source precision.

`HV` is the task-reference-normalized hypervolume used by the reported statistics. `HV_AUC` is the recorded evaluation-based hypervolume area. `IGD_plus_raw` is the original distance, `IGD_plus_range_normalized` is the normalized distance d, and `IGD_range_utility` is the recorded utility 1/(1+d). Empty-front outcomes retain the original zero quality/utility and infinite distance values. Aggregated HV and convergence-display reference-front normalization follow the definitions in the paper and supplementary material.

Each task receives equal weight in overall summaries. Summary rows retain their explicit scope, metric, contrast and multiplicity-correction labels; select the intended scope rather than averaging different summaries together. Formal and parameter matrices are separate studies; the parameter default reuses the corresponding accepted formal results. The timing summary covers 486 timing runs on the three stated probes, with six observations per method/probe cell.

The eight parameter controls use these lower/default/upper settings: initial prediction threshold 0.60/0.80/0.90; allocation update weight 0.125/0.25/0.50; reserved search mass 0.275/0.55/0.825; BFMS step and neighbor multipliers 0.5/1/2; allocation temperature 0.175/0.35/0.70; basin count 6/12/24; repair depth 1/3/5. Population/initialization settings retain their recorded role labels and columns. The default uses population 64 and 128 initial evaluations.

## Reading and checking

```python
import csv
from pathlib import Path
with Path('runs/formal118-40seeds.csv').open(encoding='utf-8', newline='') as stream:
    rows = csv.DictReader(stream)
    first = next(rows)
    print(first['case'], first['seed'], first['configuration'], first['HV'])
```

The export was checked cell by cell against accepted source arrays, with the original scientific definitions, identifiers and measurements preserved. Source filesystem locations, execution-service configuration and algorithm implementation are excluded from this numerical dataset. Algorithm distribution and benchmark definitions are maintained in the repository root.
