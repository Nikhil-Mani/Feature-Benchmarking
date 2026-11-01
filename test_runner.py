"""Integration checks using synthetic videos; no upstream clones or model downloads."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

RUNNER = Path(__file__).with_name("run_features.py")


class RunnerIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def video(self, blank=False):
        path = self.root / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (320, 240))
        if not writer.isOpened():
            self.skipTest("MJPG video codec unavailable")
        base = np.random.default_rng(42).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (3, 3), 0)
        try:
            for i in range(8):
                frame = (np.zeros_like(base) if blank else
                         cv2.warpAffine(base, np.float32([[1, 0, i], [0, 1, i * .5]]), (320, 240)))
                writer.write(frame)
        finally:
            writer.release()
        return path

    def invoke(self, path, output, *extra):
        return subprocess.run([sys.executable, str(RUNNER), str(path), "--output", str(output),
                               "--algorithms", "klt", "orb", *extra],
                              capture_output=True, text=True, timeout=30)

    def test_same_sampled_pairs_and_exports(self):
        output = self.root / "results"
        result = self.invoke(self.video(), output, "--frame-step", "2", "--max-frames", "4", "--max-size", "256")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "summary.json").read_text())
        for name in ("klt", "orb"):
            self.assertEqual(report["algorithms"][name]["pairs"], 3)
            with (output / name / "metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([(r["frame0"], r["frame1"]) for r in rows], [("0", "2"), ("2", "4"), ("4", "6")])
            points = np.load(output / name / "matches" / "00000000_00000002.npz")
            self.assertGreater(len(points["points0"]), 0)
            self.assertEqual(points["points0"].shape, points["points1"].shape)
            self.assertEqual(len(points["inliers"]), len(points["points0"]))
            self.assertTrue(np.isfinite(points["points0"]).all())
            self.assertTrue((points["points0"][:, 0] < 256).all())
            self.assertTrue((output / name / "previews" / "00000000_00000002.png").is_file())

    def test_blank_video_has_zero_matches(self):
        output = self.root / "blank"
        result = self.invoke(self.video(blank=True), output, "--max-frames", "2", "--previews", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "summary.json").read_text())
        for name in ("klt", "orb"):
            self.assertEqual(report["algorithms"][name]["mean_matches"], 0)
            self.assertIsNone(report["algorithms"][name]["median_frontend_ms_excluding_first"])

    def test_invalid_video_does_not_create_output(self):
        output = self.root / "absent"
        result = self.invoke(self.root / "missing.mp4", output)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("preserve me")
        result = self.invoke(self.video(), output)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(sentinel.read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
