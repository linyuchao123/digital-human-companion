import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from services.avatar.evaluation import (
    emotion_csv_path,
    inspect_evaluation_assets,
    load_evaluation_samples,
)


class AvatarEvaluationTests(unittest.TestCase):
    def test_official_pairs_expand_forward_then_reverse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index_csv = Path(temp_dir) / "person_specific_val.csv"
            with index_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["", "speaker_path", "listener_path"])
                writer.writerow([1, "NoXI/session/Expert_video/1", "NoXI/session/Novice_video/1"])

            samples = load_evaluation_samples(index_csv)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].direction, "forward")
        self.assertEqual(samples[1].direction, "reverse")
        self.assertEqual(samples[1].speaker_path, samples[0].listener_path)

    def test_recola_person_id_maps_to_emotion_role(self):
        path = emotion_csv_path(
            Path("/validation"), "RECOLA/group-2/P41/3"
        )

        self.assertEqual(
            path,
            Path("/validation/Emotion/RECOLA/group-2/P1/3.csv"),
        )

    def test_preflight_reports_wrong_neighbor_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_csv = root / "index.csv"
            with index_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["", "speaker_path", "listener_path"])
                writer.writerow([1, "NoXI/session/Expert_video/1", "NoXI/session/Novice_video/1"])
            for role in ("P1", "P2"):
                path = root / "val" / "Emotion" / "NoXI" / "session" / role / "1.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(
                    path,
                    np.zeros((4, 25), dtype=np.float32),
                    delimiter=",",
                    header=",".join(f"f{i}" for i in range(25)),
                    comments="",
                )
            matrix = root / "neighbors.npy"
            np.save(matrix, np.ones((1, 1), dtype=np.float32))

            report = inspect_evaluation_assets(
                val_root=root / "val",
                index_csv=index_csv,
                neighbor_matrix=matrix,
                target_len=4,
                require_checkpoint=False,
            )

        self.assertFalse(report.ready)
        self.assertIn("邻接矩阵形状应为", report.errors[0])

    def test_preflight_reports_zero_byte_emotion_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_csv = root / "index.csv"
            with index_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["", "speaker_path", "listener_path"])
                writer.writerow([1, "NoXI/session/Expert_video/1", "NoXI/session/Novice_video/1"])
            for role in ("P1", "P2"):
                path = root / "val" / "Emotion" / "NoXI" / "session" / role / "1.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            matrix = root / "neighbors.npy"
            np.save(matrix, np.ones((2, 2), dtype=np.float32))

            report = inspect_evaluation_assets(
                val_root=root / "val",
                index_csv=index_csv,
                neighbor_matrix=matrix,
                target_len=4,
                require_checkpoint=False,
            )

        self.assertFalse(report.ready)
        self.assertEqual(len(report.invalid_files), 2)
        self.assertTrue(all("文件为空" in item for item in report.invalid_files))
        self.assertEqual(report.to_summary_dict()["invalid_file_count"], 2)


if __name__ == "__main__":
    unittest.main()
