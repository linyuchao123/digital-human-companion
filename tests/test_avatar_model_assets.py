import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.avatar.model_assets import (
    PROJECT_ROOT,
    inspect_checkpoint,
    resolve_checkpoint_path,
)


class AvatarModelAssetTests(unittest.TestCase):
    def test_missing_checkpoint_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status = inspect_checkpoint(Path(temp_dir) / "missing.pt")

        self.assertFalse(status.exists)
        self.assertFalse(status.ready)
        self.assertEqual(status.integrity, "missing")

    def test_matching_checkpoint_passes_integrity_check(self):
        content = b"digital-xinyu-model"
        expected_sha256 = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "best_model.pt"
            checkpoint.write_bytes(content)
            status = inspect_checkpoint(
                checkpoint,
                expected_sha256=expected_sha256,
                expected_size=len(content),
            )

        self.assertTrue(status.exists)
        self.assertTrue(status.ready)
        self.assertEqual(status.integrity, "verified")
        self.assertEqual(status.sha256, expected_sha256)

    def test_size_mismatch_stops_before_checksum(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "best_model.pt"
            checkpoint.write_bytes(b"incomplete")
            status = inspect_checkpoint(
                checkpoint,
                expected_sha256="0" * 64,
                expected_size=1024,
            )

        self.assertFalse(status.ready)
        self.assertEqual(status.integrity, "size_mismatch")
        self.assertIsNone(status.sha256)

    def test_relative_environment_path_resolves_from_project_root(self):
        with patch.dict(
            os.environ,
            {"FACE_DRIVER_CHECKPOINT": "models/custom.pt"},
            clear=False,
        ):
            resolved = resolve_checkpoint_path()

        self.assertEqual(resolved, PROJECT_ROOT / "models" / "custom.pt")


if __name__ == "__main__":
    unittest.main()
