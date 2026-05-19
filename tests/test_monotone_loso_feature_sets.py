"""Feature-set wiring tests for monotone LOSO calibration."""

from pathlib import Path

from bayescatrack.association.activity_similarity import ACTIVITY_TIEBREAKER_FEATURES
from bayescatrack.association.calibrated_costs import (
    LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    SPLIT_ROI_STAT_FEATURES,
)
from bayescatrack.association.monotone_ranker import MonotoneRankerOptions
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_monotone_loso_calibration import (
    config_with_feature_dependencies,
    monotone_feature_names_for_set,
    monotone_options_for_feature_schema,
)


def test_full_monotone_feature_set_includes_split_local_and_activity_features() -> None:
    feature_names = monotone_feature_names_for_set("full")

    assert set(SPLIT_ROI_STAT_FEATURES).issubset(feature_names)
    assert set(LOCAL_EVIDENCE_ASSOCIATION_FEATURES).issubset(feature_names)
    assert set(ACTIVITY_TIEBREAKER_FEATURES).issubset(feature_names)
    assert "roi_feature_cost" not in feature_names


def test_local_evidence_feature_set_enables_component_production() -> None:
    config = Track2pBenchmarkConfig(
        data=Path("/tmp/track2p"),
        method="global-assignment",
        cost="calibrated",
    )

    updated = config_with_feature_dependencies(
        config, monotone_feature_names_for_set("local-evidence")
    )

    assert updated.pairwise_cost_kwargs is not None
    assert updated.pairwise_cost_kwargs["local_evidence_components"] is True
    assert config.pairwise_cost_kwargs is None


def test_activity_only_feature_set_does_not_force_local_evidence_components() -> None:
    config = Track2pBenchmarkConfig(
        data=Path("/tmp/track2p"),
        method="global-assignment",
        cost="calibrated",
        pairwise_cost_kwargs={"similarity_epsilon": 1.0e-5},
    )

    updated = config_with_feature_dependencies(
        config, monotone_feature_names_for_set("activity-tiebreaker")
    )

    assert updated is config
    assert updated.pairwise_cost_kwargs == {"similarity_epsilon": 1.0e-5}


def test_default_monotone_options_use_selected_cost_like_features() -> None:
    options = monotone_options_for_feature_schema(
        MonotoneRankerOptions(), monotone_feature_names_for_set("full")
    )

    assert "activity_tiebreaker_cost" in options.monotone_feature_names
    assert "activity_tiebreaker_missing" in options.monotone_feature_names
    assert "activity_tiebreaker_available" not in options.monotone_feature_names
    assert "fluorescence_similarity_available" not in options.monotone_feature_names
    assert "missing_stat_indicator" in options.monotone_feature_names
