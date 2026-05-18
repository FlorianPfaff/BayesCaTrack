import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    NamedPairwiseFeatureSchema,
    pairwise_components_from_bundle,
    pairwise_feature_schema,
    pairwise_feature_tensor,
    with_session_gap_component,
)
from bayescatrack.core.bridge import SessionAssociationBundle


def _state_covariances_from_position_covariances(
    position_covariances: np.ndarray,
) -> np.ndarray:
    state_covariances = np.zeros((4, 4, position_covariances.shape[2]), dtype=float)
    state_covariances[0, 0, :] = position_covariances[0, 0, :]
    state_covariances[0, 2, :] = position_covariances[0, 1, :]
    state_covariances[2, 0, :] = position_covariances[1, 0, :]
    state_covariances[2, 2, :] = position_covariances[1, 1, :]
    return state_covariances


def _association_bundle(
    position_covariances: np.ndarray, measurement_covariances: np.ndarray
) -> SessionAssociationBundle:
    n_reference = position_covariances.shape[2]
    n_measurement = measurement_covariances.shape[2]
    shape = (n_reference, n_measurement)
    return SessionAssociationBundle(
        reference_session_name="ref",
        measurement_session_name="meas",
        reference_state_means=np.zeros((4, n_reference), dtype=float),
        reference_state_covariances=_state_covariances_from_position_covariances(
            position_covariances
        ),
        measurements=np.zeros((2, n_measurement), dtype=float),
        measurement_covariances=measurement_covariances,
        measurement_matrix=np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=float
        ),
        pairwise_cost_matrix=np.zeros(shape, dtype=float),
        reference_roi_indices=np.arange(n_reference, dtype=int),
        measurement_roi_indices=np.arange(n_measurement, dtype=int),
        pairwise_components={
            "centroid_distance": np.zeros(shape, dtype=float),
            "iou": np.ones(shape, dtype=float),
            "mask_cosine_similarity": np.ones(shape, dtype=float),
            "area_ratio_cost": np.zeros(shape, dtype=float),
            "roi_feature_cost": np.zeros(shape, dtype=float),
            "cell_probability_cost": np.zeros(shape, dtype=float),
        },
    )


def test_pairwise_components_from_bundle_adds_covariance_shape_features():
    reference_covariances = np.stack(
        [np.diag([4.0, 1.0])],
        axis=-1,
    )
    measurement_covariances = np.stack(
        [np.diag([4.0, 1.0]), np.diag([1.0, 4.0]), np.diag([8.0, 2.0])],
        axis=-1,
    )
    components = pairwise_components_from_bundle(
        _association_bundle(reference_covariances, measurement_covariances)
    )

    assert components["covariance_shape_cost"].shape == (1, 3)
    assert components["covariance_logdet_cost"].shape == (1, 3)
    assert components["covariance_shape_similarity"].shape == (1, 3)
    npt.assert_allclose(components["covariance_shape_cost"][0, 0], 0.0, atol=1.0e-12)
    npt.assert_allclose(components["covariance_logdet_cost"][0, 0], 0.0, atol=1.0e-12)
    assert (
        components["covariance_shape_similarity"][0, 0]
        > components["covariance_shape_similarity"][0, 1]
    )
    assert (
        components["covariance_logdet_cost"][0, 2]
        > components["covariance_logdet_cost"][0, 0]
    )


def test_default_calibrated_features_include_covariance_shape_components():
    reference_covariances = np.stack([np.diag([4.0, 1.0])], axis=-1)
    measurement_covariances = np.stack(
        [np.diag([4.0, 1.0]), np.diag([1.0, 4.0])], axis=-1
    )
    components = pairwise_components_from_bundle(
        _association_bundle(reference_covariances, measurement_covariances)
    )
    features = pairwise_feature_tensor(
        components, feature_names=DEFAULT_ASSOCIATION_FEATURES
    )

    assert "covariance_shape_cost" in DEFAULT_ASSOCIATION_FEATURES
    assert "covariance_logdet_cost" in DEFAULT_ASSOCIATION_FEATURES
    assert features.shape == (1, 2, len(DEFAULT_ASSOCIATION_FEATURES))
    assert np.all(np.isfinite(features))


def test_default_calibrated_features_include_session_gap_component():
    reference_covariances = np.stack([np.diag([4.0, 1.0])], axis=-1)
    measurement_covariances = np.stack(
        [np.diag([4.0, 1.0]), np.diag([1.0, 4.0])], axis=-1
    )
    components = pairwise_components_from_bundle(
        _association_bundle(reference_covariances, measurement_covariances),
        session_gap=2,
    )
    features = pairwise_feature_tensor(
        components, feature_names=DEFAULT_ASSOCIATION_FEATURES
    )
    session_gap_index = DEFAULT_ASSOCIATION_FEATURES.index("session_gap")

    assert "session_gap" in DEFAULT_ASSOCIATION_FEATURES
    npt.assert_allclose(components["session_gap"], np.full((1, 2), 2.0))
    npt.assert_allclose(features[:, :, session_gap_index], np.full((1, 2), 2.0))


def test_pairwise_feature_schema_uses_named_schema_with_domain_transforms():
    components = {
        "iou": np.array([[0.8, 0.25]], dtype=float),
        "session_gap": np.array([[2.0, 2.0]], dtype=float),
    }
    schema = pairwise_feature_schema(("one_minus_iou", "session_gap"))
    features = np.asarray(schema.build_tensor(components), dtype=float)

    assert isinstance(schema, NamedPairwiseFeatureSchema)
    assert schema.feature_names == ("one_minus_iou", "session_gap")
    npt.assert_allclose(features[:, :, 0], np.array([[0.2, 0.75]]))
    npt.assert_allclose(features[:, :, 1], np.array([[2.0, 2.0]]))


def test_session_gap_component_requires_positive_gap():
    with pytest.raises(ValueError, match="session_gap must be positive"):
        with_session_gap_component(
            {"centroid_distance": np.zeros((1, 1))}, session_gap=0
        )
