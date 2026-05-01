from __future__ import annotations

import csv
import sys
import types

import numpy as np

from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_calibration_export import export_loso_calibration_csv
from tests.test_track2p_benchmark import _write_ground_truth_csv, _write_subject


def _install_fake_logistic_association_model(monkeypatch):
    fake_association_models = types.ModuleType("pyrecest.utils.association_models")

    class LogisticPairwiseAssociationModel:
        def __init__(self, **_kwargs):
            self.n_features_in_ = None

        def fit(self, features, labels, sample_weight=None):
            del labels, sample_weight
            features = np.asarray(features, dtype=float)
            self.n_features_in_ = features.shape[-1]
            return self

        def predict_match_probability(self, features):
            features = np.asarray(features, dtype=float)
            score = -np.sum(features, axis=-1)
            return 1.0 / (1.0 + np.exp(-score))

        def pairwise_cost_matrix(self, features):
            probabilities = np.clip(self.predict_match_probability(features), 1.0e-12, 1.0 - 1.0e-12)
            return -np.log(probabilities)

    fake_association_models.LogisticPairwiseAssociationModel = LogisticPairwiseAssociationModel
    monkeypatch.setitem(sys.modules, "pyrecest.utils.association_models", fake_association_models)


def test_loso_calibration_export_writes_out_of_fold_pairwise_rows(tmp_path, monkeypatch, write_raw_npy_session):
    for subject_name in ("jm001", "jm002"):
        subject_dir = tmp_path / subject_name
        _write_subject(subject_dir, write_raw_npy_session, write_reference=False)
        _write_ground_truth_csv(
            subject_dir,
            ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a"),
            ((0, 0, 0), (1, 1, 1)),
        )

    _install_fake_logistic_association_model(monkeypatch)

    from bayescatrack.association import calibrated_costs

    monkeypatch.setattr(calibrated_costs, "register_plane_pair", lambda _reference, moving, **_kwargs: moving)

    output_path = tmp_path / "results" / "calibration.csv"
    written = export_loso_calibration_csv(
        Track2pBenchmarkConfig(
            data=tmp_path,
            method="global-assignment",
            split="leave-one-subject-out",
            cost="calibrated",
            max_gap=2,
        ),
        output_path,
    )

    assert written > 0
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == written
    assert {row["held_out_subject"] for row in rows} == {"jm001", "jm002"}
    assert {row["subject"] for row in rows} == {"jm001", "jm002"}
    assert {row["session_gap"] for row in rows} == {"1", "2"}
    assert {row["label"] for row in rows} <= {"0", "1"}
    assert all(0.0 <= float(row["p_same"]) <= 1.0 for row in rows)
    assert all(float(row["link_cost"]) >= 0.0 for row in rows)
    assert "feature_centroid_distance" in rows[0]
    assert "feature_one_minus_iou" in rows[0]
    assert "feature_covariance_shape_cost" in rows[0]
