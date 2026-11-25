# Results

All values are ATE RMSE in metres (SE(3) alignment without scale correction, full sequence), produced by
`notebooks/depth_gate_experiments.ipynb` in one fixed environment. "Criterion" is b/d ≥ θ with kσ_px = 2.5 px.

| File | Content |
|---|---|
| `runs_three_starts.csv` | The 153 runs behind Table 1: strict 1° threshold, loose 0.3° threshold, and loose threshold with the criterion, for 17 sequences and start delays of 0, 0.1 and 0.2 s |
| `table2_threshold_sweep.csv` | Table 2 with its supporting columns: seven thresholds from 0° to 1°, without (`alone`) and with the criterion, 51 runs per entry |
| `external/openvins_check.csv` | Monocular OpenVINS (commit 6948812) on the 11 EuRoC sequences from three starts, with its default bound of 40 on the ratio of landmark distance to baseline and with the derived bound of 184 |
| `full/` | The complete output of section 7 and 8 of the notebook (per-run summary of all 816 runs, every table as CSV and Markdown, `paper_numbers.txt`) |

`full/summary_all_runs.csv` has one row per run with the configuration, the ATE, the counters of the admission chain
(`n_tried`, `rej_parallax`, …, `n_gate_rejected`, `n_used`) and the timing. Placing it in the notebook's `results/`
folder lets the notebook rebuild the ATE tables without re-running the filter.
