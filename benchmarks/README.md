# Track2p benchmark manifests

This directory contains reproducible benchmark-suite definitions for the
paper-facing Track2p comparison.

## Default ablation suite

`track2p_default_ablations.json` is intended as the first full quantitative
table once the raw Suite2p ROI index space and manual `ground_truth.csv` files
are available. It covers the rows needed to separate the main hypotheses:

- Track2p default output.
- Registered-IoU global assignment with consecutive edges only (`gap1`).
- Registered-IoU global assignment with skip edges (`gap2`).
- LOSO solver-prior tuning for registered-IoU, ROI-aware, and shifted ROI-aware costs.
- LOSO calibrated logistic costs with default features.
- LOSO calibrated logistic costs with default plus local-evidence features.
- LOSO monotone hard-negative ranker costs.

The manifest assumes the benchmark data live at `../data/track2p` relative to
this directory and that manual ground-truth CSV files are discoverable per
subject. Edit the `defaults.data` and, if necessary, `defaults.reference` fields
for a different local layout.

Run the full suite with:

```bash
python -m bayescatrack benchmark suite \
  benchmarks/track2p_default_ablations.json \
  --output-dir results/track2p_default_ablations \
  --summary-format table
```

The defaults intentionally set `include_non_cells=true`, because the manual
Track2p ground truth may refer to raw Suite2p `stat.npy` row indices rather than
to a post-filtered cell-only ROI index space.

The solver-prior and monotone-ranker rows use the extended manifest `runner`
field; ordinary benchmark rows continue to use the default `track2p` runner.
