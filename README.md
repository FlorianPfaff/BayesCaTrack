# BayesCaTrack

BayesCaTrack is a recursive Bayesian cell tracking toolkit for Track2p-style and related calcium-imaging datasets. It focuses on Track2p / Suite2p data ingestion, ROI representation, registration-aware association costs, Track2p reference evaluation, PyRecEst-ready exports, and reproducible Track2p benchmark ablations.

## Package layout

The package now has a single public source namespace:

```text
bayescatrack/
  src/
    bayescatrack/
      core/
      association/
      datasets/
        track2p/
      evaluation/
      experiments/
      io/
        suite2p.py
        track2p.py
  tests/
```

## What it gives you

- Loads Track2p-style subject/session directories.
- Reconstructs Suite2p ROI masks from `stat.npy`.
- Computes ROI centroids and spatial covariance matrices.
- Builds constant-velocity state moments that can initialize PyRecEst filters.
- Builds ROI-aware pairwise association costs and standard `SessionAssociationBundle` objects.
- Registers later-session ROIs into an earlier session's coordinate frame before association.
- Stitches registration-aware pairwise matches into full predicted track rows.
- Loads Track2p reference identities and scores pairwise association predictions.
- Runs Track2p baseline, PyRecEst global-assignment, and LOSO calibrated-cost benchmark ablations.
- Exports per-session measurements and state moments to a single `.npz` archive.
- Includes a CLI for quick inspection.

## Architecture boundary

BayesCaTrack owns calcium-imaging semantics such as Track2p/Suite2p ingestion, ROI representation, ROI-derived association costs, registration-aware ROI warping, and Track2p benchmark scoring. PyRecEst is the backend for generic recursive Bayesian estimation primitives such as distributions, filters, point-set registration, and multi-session assignment.

See [docs/architecture.md](docs/architecture.md) for the detailed boundary and the Track2p outperformance workstream.

## CLI examples

Inspect one subject:

```bash
python -m bayescatrack summary /path/to/jm039 --plane plane0
```

Export PyRecEst-ready states:

```bash
python -m bayescatrack export /path/to/jm039 /tmp/jm039_plane0.npz \
  --plane plane0 \
  --input-format auto \
  --weighted-masks \
  --weighted \
  --velocity-variance 25.0
```

Validate that PyRecEst objects can be instantiated during export:

```bash
python -m bayescatrack export /path/to/jm039 /tmp/jm039_plane0.npz \
  --validate-pyrecest
```

Run the Track2p default benchmark row against independent manual ground truth:

```bash
python -m bayescatrack benchmark track2p \
  --data /path/to/track2p_zenodo \
  --method track2p-baseline \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt
```

If a subject directory contains `ground_truth.csv`, the benchmark can use it as
the reference automatically. You can also point `--reference` at a
`ground_truth.csv` file or at a separate ground-truth root and declare
`--reference-kind manual-gt`. Ground-truth ROI indices are validated against the
loaded Suite2p ROI indices, so references that use raw `stat.npy` row indices
will fail clearly if Suite2p cell filtering removed any referenced ROI; use
`--include-non-cells` when benchmarking against all `stat.npy` rows.

The benchmark refuses Track2p outputs and already row-aligned Suite2p rows as
references by default because those are not independent evidence for a
paper-facing comparison. For plumbing checks only, pass
`--allow-track2p-as-reference-for-smoke-test`. Sparse manual ground truth is
handled by scoring only predicted tracks whose seed-session ROI appears in the
reference seed set; this avoids counting unlabelled cells as false positives.

Create a small synthetic Suite2p-style subject for benchmark development:

```python
from bayescatrack.datasets.track2p import (
    SyntheticFalsePositiveRoi,
    SyntheticTrack2pSubjectConfig,
    write_synthetic_track2p_subject,
)

generated = write_synthetic_track2p_subject(
    "/tmp/bayescatrack-synthetic",
    SyntheticTrack2pSubjectConfig(
        subject_name="jm_synthetic",
        missing_detections=((1, 2),),
        non_cell_tracks=(0,),
        false_positive_rois=(
            SyntheticFalsePositiveRoi(session_index=1, center_yx=(15.0, 15.0)),
        ),
    ),
)
```

Run the clean global-assignment ablation with registered IoU costs and skip edges:

```bash
python -m bayescatrack benchmark track2p \
  --data /path/to/track2p_zenodo \
  --method global-assignment \
  --cost registered-iou \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt \
  --max-gap 2
```

Run the BayesCaTrack ROI-aware cost ablation:

```bash
python -m bayescatrack benchmark track2p \
  --data /path/to/track2p_zenodo \
  --method global-assignment \
  --cost roi-aware \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt \
  --max-gap 2
```

Run the LOSO calibrated-cost ablation:

```bash
python -m bayescatrack benchmark track2p \
  --data /path/to/track2p_zenodo \
  --method global-assignment \
  --cost calibrated \
  --split leave-one-subject-out \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt \
  --max-gap 2
```

The benchmark prints a compact table by default and can also write JSON or CSV via `--format json --output results.json` or `--format csv --output results.csv`.

Run a reproducible benchmark suite from one JSON manifest:

```json
{
  "defaults": {
    "data": "data/jm039",
    "input_format": "suite2p",
    "include_behavior": false,
    "include_non_cells": true
  },
  "runs": [
    {
      "name": "track2p-default",
      "method": "track2p-baseline",
      "output": "results/track2p_default.csv"
    },
    {
      "name": "registered-iou",
      "method": "global-assignment",
      "cost": "registered-iou",
      "max_gap": 2,
      "output": "results/registered_iou.csv"
    }
  ],
  "comparisons": [
    {
      "name": "summary",
      "inputs": {
        "Track2p default": "track2p-default",
        "BayesCaTrack registered IoU": "registered-iou"
      },
      "output": "results/comparison.md"
    }
  ]
}
```

```bash
python -m bayescatrack benchmark suite benchmarks.json --summary-format table
```

Manifest paths are resolved relative to the manifest file. This makes it possible to keep the same benchmark definition for synthetic fixtures, the current Zenodo-aligned data, and the future pre-Track2p Suite2p folders once they are available.

## Python example

```python
from bayescatrack import load_track2p_subject

sessions = load_track2p_subject("/path/to/jm039", plane_name="plane0", input_format="auto")
first_session = sessions[0].plane_data

measurements = first_session.to_measurement_matrix(order="xy")
means, covariances = first_session.to_constant_velocity_state_moments(
    order="xy",
    velocity_variance=25.0,
)

# Requires PyRecEst to be importable.
filters = first_session.to_pyrecest_kalman_filters(
    order="xy",
    velocity_variance=25.0,
)
```

## Registration example

```python
from bayescatrack import load_track2p_subject
from bayescatrack.registration import build_registered_session_pair_association_bundle

sessions = load_track2p_subject("/path/to/jm039", plane_name="plane0")
registered = build_registered_session_pair_association_bundle(
    sessions[0],
    sessions[1],
    registration_model="affine",
    pairwise_cost_kwargs={
        "max_centroid_distance": 25.0,
        "roi_feature_weight": 0.25,
    },
)

pairwise_cost_matrix = registered.association_bundle.pairwise_cost_matrix
registered_plane = registered.plane_registration.registered_measurement_plane
```

## Tracking Runner Example

```python
from bayescatrack.tracking import run_registered_subject_tracking

result = run_registered_subject_tracking(
    "/path/to/jm039",
    plane_name="plane0",
    input_format="auto",
    registration_model="affine",
    pairwise_cost_kwargs={
        "max_centroid_distance": 25.0,
        "roi_feature_weight": 0.25,
    },
    assignment_max_cost=50.0,
)

predicted_tracks = result.track_rows
link_costs = result.link_costs
summary = result.score_summary()
```

## Reference example

```python
from bayescatrack.reference import load_track2p_reference, score_pairwise_matches

reference = load_track2p_reference("/path/to/jm039/track2p", plane_name="plane0")
reference_pairs = reference.pairwise_matches(0, 1, curated_only=True)
scores = score_pairwise_matches(predicted_pairs, reference_pairs)
```

## Notes

- The state layout is `[pos_1, vel_1, pos_2, vel_2]`.
- The benchmark keeps Track2p/Suite2p and calcium-imaging assumptions inside BayesCaTrack while using PyRecEst only for abstract global assignment.
- BayesCaTrack is the canonical import namespace for the package.
- `--validate-pyrecest` is useful when you want the export step to fail early if the current environment cannot instantiate the expected PyRecEst classes.
