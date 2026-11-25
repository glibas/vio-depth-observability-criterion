# A Depth-Observability Admission Criterion for UAV Visual-Inertial Odometry

Code, notebook and results for the paper

> G. Briia, I. Turkin, O. Liubimov. *A Depth-Observability Admission Criterion for UAV Visual-Inertial Odometry.*
> ISEAS / ISUN-AI 2026.

## The criterion

A landmark is admitted to the filter update only if the camera displacement over its track, `b`, divided by the
triangulated depth, `d`, exceeds a bound derived from the tracking noise:

    b / d  >=  theta,     theta = k * sigma_px / f,     k * sigma_px = 2.5 px

which gives theta = 0.0055 for the EuRoC pinhole camera and 0.0131 for the TUM-VI fisheye camera. It replaces part of
the work of the usual apparent-parallax threshold, so that this threshold can be relaxed or dropped.

## What is here

    msckf/            monocular MSCKF (configuration, dataset loaders, KLT tracker, filter, evaluation)
    experiments/      run_sequence.py    one sequence with one admission setting
                      failure_cases.py   synthetic tracks: where each one leaves the admission chain
                      gate_cost.py       cost of the test against one triangulation
    charts/           make_figures.py    MH_03 position-error illustration
    notebooks/        depth_gate_experiments.ipynb   runs everything and writes every table of the paper
    tools/            build_notebook.py  packs msckf/, experiments/ and charts/ into the notebook
    results/          per-run results and the tables of the paper (see results/README.md)

In the code the criterion is called the *gate* (`--gate ratio`), and its variant with the baseline component
orthogonal to the viewing ray is `--gate perp`.

## Main results

17 sequences (EuRoC MAV: 11, TUM-VI room1–room6), each run from three start times (51 runs per setting):

- The 0.3° threshold with the criterion is more accurate than the strict 1° threshold in 45 of 51 runs and in 15 of 17
  sequence medians (Wilcoxon p = 0.0003); the median ATE is 0.096 m against 0.157 m.
- Most of this gain comes from relaxing the threshold. Over seven thresholds from 0° to 1°, runs with and without the
  criterion do not differ significantly (p ≥ 0.15).
- The criterion acts as a safeguard: without it MH_03 exceeds 1 m at four thresholds (0.1° to 0.5°), with it MH_03
  stays below 0.25 m in all 21 runs. With a threshold of 0.2° or less, or none, it causes no failure of its own;
  stacked on a 0.3° to 0.7° threshold it fails in three runs at start-up.
- The test takes 0.34 ms of a 74.4 ms mean frame time (0.46%).
- In monocular OpenVINS, replacing the default bound of 40 by the derived 184 gave no accuracy benefit.

## Reproduce

The notebook is self-contained (the code of this repository is embedded in it) and runs on Google Colab or in a local
Jupyter.

1. Download the datasets from their pages (they are not redistributed here):
   EuRoC MAV, https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets, and
   TUM-VI (`dataset-room1_512_16` … `dataset-room6_512_16`), https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset.
   Put them in one folder; ASL folders, zip/tar archives and ROS1 bags are recognised.
2. Open `notebooks/depth_gate_experiments.ipynb`, set `DRIVE_ROOT` (the folder that contains `Datasets/`), choose the
   tiers and run all cells. Runs and tables are written to `DRIVE_ROOT/depth-observability-criterion/`.

All 816 runs take about 7 h on an 8-core machine (4–6 CPU-minutes per run); tiers A and C (170 runs) are enough for
Table 1. Every run is deterministic and single-threaded, and finished runs are cached and skipped.

A single run from the command line:

    pip install -r requirements.txt
    python experiments/run_sequence.py <sequence dir> --gate ratio --gate-value 0.0055 --parallax 0.3 --out run.json

After changing the code, rebuild the notebook with `python tools/build_notebook.py . notebooks/depth_gate_experiments.ipynb`.

## Other estimators

The OpenVINS and VINS-Mono checks quoted in the paper were run with the upstream code bases outside this repository;
`results/external/` holds the OpenVINS per-run values.

## Citation

Please cite the paper above. Citation metadata is in `CITATION.cff`.

## Licence

MIT, see `LICENSE`. The EuRoC MAV and TUM-VI datasets are the property of their authors and have their own licences.

## Funding

This research is carried out at the expense of the general fund of the state budget under the title "Methods and
means of ensuring fault tolerance of on-board hardware and software systems of UAV and nanosatellite computing
platforms", state registration number 0126U002262.
