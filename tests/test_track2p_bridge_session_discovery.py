import numpy as np

from bayescatrack.core.bridge import find_track2p_session_dirs


def test_find_track2p_session_dirs_ignores_date_dirs_without_track2p_data(tmp_path):
    subject_dir = tmp_path / "jm038"
    subject_dir.mkdir()
    (subject_dir / "2024-05-01_a").mkdir()
    (subject_dir / "2024-05-02_a" / "suite2p" / "plane0").mkdir(parents=True)
    (subject_dir / "notes").mkdir()

    session_names = [
        session_dir.name for session_dir in find_track2p_session_dirs(subject_dir)
    ]

    assert session_names == ["2024-05-02_a"]


def test_find_track2p_session_dirs_can_require_suite2p_plane_data(tmp_path):
    subject_dir = tmp_path / "jm038"
    subject_dir.mkdir()
    incomplete = subject_dir / "2024-05-01_a" / "suite2p" / "plane0"
    incomplete.mkdir(parents=True)
    complete = subject_dir / "2024-05-02_a" / "suite2p" / "plane0"
    complete.mkdir(parents=True)
    np.save(
        complete / "stat.npy",
        np.asarray(
            [
                {
                    "ypix": np.asarray([0], dtype=int),
                    "xpix": np.asarray([0], dtype=int),
                }
            ],
            dtype=object,
        ),
    )
    wrong_plane = subject_dir / "2024-05-03_a" / "suite2p" / "plane1"
    wrong_plane.mkdir(parents=True)
    np.save(wrong_plane / "stat.npy", np.asarray([], dtype=object))

    session_names = [
        session_dir.name
        for session_dir in find_track2p_session_dirs(
            subject_dir,
            plane_name="plane0",
            input_format="suite2p",
        )
    ]

    assert session_names == ["2024-05-02_a"]


def test_find_track2p_session_dirs_can_require_raw_npy_plane_data(tmp_path):
    subject_dir = tmp_path / "jm038"
    subject_dir.mkdir()
    incomplete = subject_dir / "2024-05-01_a" / "data_npy" / "plane0"
    incomplete.mkdir(parents=True)
    np.save(incomplete / "rois.npy", np.zeros((1, 2, 2), dtype=bool))

    complete = subject_dir / "2024-05-02_a" / "data_npy" / "plane0"
    complete.mkdir(parents=True)
    np.save(complete / "rois.npy", np.zeros((1, 2, 2), dtype=bool))
    np.save(complete / "F.npy", np.zeros((1, 3), dtype=float))
    np.save(complete / "fov.npy", np.zeros((2, 2), dtype=float))

    session_names = [
        session_dir.name
        for session_dir in find_track2p_session_dirs(
            subject_dir,
            plane_name="plane0",
            input_format="npy",
        )
    ]

    assert session_names == ["2024-05-02_a"]
