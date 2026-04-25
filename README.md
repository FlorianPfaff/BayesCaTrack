# BayesCaTrack

BayesCaTrack is a recursive Bayesian cell tracking toolkit for Track2p-style and related calcium-imaging datasets. It focuses on Track2p / Suite2p data ingestion, ROI representation, registration-aware association costs, Track2p reference evaluation, and PyRecEst-ready exports.

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
- Loads Track2p reference identities and scores pairwise association predictions.
- Exports per-session measurements and state moments to a single `.npz` archive.
- Includes a CLI for quick inspection.

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

## Reference example

```python
from bayescatrack.reference import load_track2p_reference, score_pairwise_matches

reference = load_track2p_reference("/path/to/jm039/track2p", plane_name="plane0")
reference_pairs = reference.pairwise_matches(0, 1, curated_only=True)
scores = score_pairwise_matches(predicted_pairs, reference_pairs)
```

## Notes

- The state layout is `[pos_1, vel_1, pos_2, vel_2]`.
- BayesCaTrack is the canonical import namespace for the package.
- `--validate-pyrecest` is useful when you want the export step to fail early if the current environment cannot instantiate the expected PyRecEst classes.
