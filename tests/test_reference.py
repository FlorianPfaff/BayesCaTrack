import numpy as np
import numpy.testing as npt

from bayescatrack.reference import pairs_from_label_vectors, score_pairwise_matches


def test_pairs_from_label_vectors_and_scores():
    reference_pairs = pairs_from_label_vectors(
        np.array([1, 2, None], dtype=object),
        np.array([2, 1, 3], dtype=object),
    )
    npt.assert_array_equal(reference_pairs, np.array([[0, 1], [1, 0]]))

    scores = score_pairwise_matches(np.array([[0, 1], [1, 2]]), reference_pairs)
    assert scores["true_positives"] == 1
    assert scores["false_positives"] == 1
    assert scores["false_negatives"] == 1
