"""Evaluation helpers for BayesCaTrack benchmarks."""

from . import calibration_diagnostics as _calibration_diagnostics
from . import complete_track_scores as _scores
from . import track2p_metrics as _track2p_metrics
from . import track_error_ledger as _track_error_ledger

brier_score = _calibration_diagnostics.brier_score
calibration_summary = _calibration_diagnostics.calibration_summary
complete_track_set = _scores.complete_track_set
expected_calibration_error = _calibration_diagnostics.expected_calibration_error
format_reliability_bin_table = _calibration_diagnostics.format_reliability_bin_table
maximum_calibration_error = _calibration_diagnostics.maximum_calibration_error
normalize_track_matrix = _scores.normalize_track_matrix
pairwise_track_set = _scores.pairwise_track_set
reliability_bin_table = _calibration_diagnostics.reliability_bin_table
score_complete_tracks = _scores.score_complete_tracks
score_false_continuations = _scores.score_false_continuations
score_pairwise_tracks = _scores.score_pairwise_tracks
score_track_matrices = _scores.score_track_matrices
summarize_track_errors = _track_error_ledger.summarize_track_errors
summarize_tracks = _scores.summarize_tracks
track_error_ledger = _track_error_ledger.track_error_ledger
track_lengths = _scores.track_lengths
score_track_matrix_against_reference = _track2p_metrics.score_track_matrix_against_reference

__all__ = list(_calibration_diagnostics.__all__) + list(_scores.__all__) + list(_track_error_ledger.__all__) + [
    "score_track_matrix_against_reference",
]
