#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALGORITHMS = ("klt", "orb", "superpoint", "aliked", "disk", "xfeat")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path, help="Input video filename")
    p.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    p.add_argument("--output", type=Path, help="New output directory (must not exist)")
    p.add_argument("--max-keypoints", type=int, default=1024)
    p.add_argument("--max-size", type=int, default=960, help="Longest image side; 0 keeps original")
    p.add_argument("--frame-step", type=int, default=1, help="Compare successive sampled frames")
    p.add_argument("--max-frames", type=int, default=300, help="Sampled frames; 0 processes full video")
    p.add_argument("--previews", type=int, default=10, help="Maximum PNG previews per algorithm")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--calibration", type=Path, help="Optional JSON with model, K, D, image_size")
    p.add_argument("--ransac-threshold", type=float, default=1.5, help="Processed-image pixels")
    p.add_argument("--fb-threshold", type=float, default=1.0, help="KLT forward/backward check, pixels")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    for name in ("max_keypoints", "frame_step"):
        if getattr(a, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if a.max_keypoints < 8:
        p.error("--max-keypoints must be at least 8")
    if a.max_size and a.max_size < 32:
        p.error("--max-size must be 0 or at least 32")
    if a.max_frames < 0 or a.max_frames == 1 or a.previews < 0:
        p.error("--max-frames must be 0 or at least 2; --previews must be nonnegative")
    if a.ransac_threshold <= 0 or a.fb_threshold <= 0:
        p.error("thresholds must be positive")
    a.algorithms = list(dict.fromkeys(a.algorithms))
    return a


def submodule(name):
    path = ROOT / "third_party" / name
    if not (path / ".git").exists():
        raise ImportError(f"Uninitialized submodule {path}. Run: git submodule update --init --recursive")
    return path


def tensor_image(image, torch, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).to(device)


class Classical:
    def __init__(self, name, a):
        self.name, self.a = name, a
        self.orb = cv2.ORB_create(nfeatures=a.max_keypoints) if name == "orb" else None

    def sync(self):
        pass

    def match(self, first, second):
        g0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        empty = np.empty((0, 2), np.float32)
        if self.name == "orb":
            k0, d0 = self.orb.detectAndCompute(g0, None)
            k1, d1 = self.orb.detectAndCompute(g1, None)
            if d0 is None or d1 is None:
                return empty, empty.copy(), len(k0), len(k1)
            matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
            return (np.array([k0[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 2),
                    np.array([k1[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 2),
                    len(k0), len(k1))
        p0 = cv2.goodFeaturesToTrack(g0, self.a.max_keypoints, 0.01, 8)
        if p0 is None:
            return empty, empty.copy(), 0, None
        kw = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        p1, status, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **kw)
        if p1 is None:
            return empty, empty.copy(), len(p0), None
        valid = status.ravel().astype(bool) & np.isfinite(p1.reshape(-1, 2)).all(axis=1)
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            return empty, empty.copy(), len(p0), None
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1[indices], None, **kw)
        if back is None:
            return empty, empty.copy(), len(p0), None
        error = np.linalg.norm(p0[indices].reshape(-1, 2) - back.reshape(-1, 2), axis=1)
        q = p1[indices].reshape(-1, 2)
        height, width = g1.shape
        good = (back_status.ravel().astype(bool) & np.isfinite(error)
                & (error <= self.a.fb_threshold) & (q[:, 0] >= 0) & (q[:, 0] < width)
                & (q[:, 1] >= 0) & (q[:, 1] < height))
        return p0[indices].reshape(-1, 2)[good], q[good], len(p0), None


class Learned:
    def __init__(self, name, a):
        import torch
        self.torch, self.name, self.a = torch, name, a
        self.device = "cuda" if a.device == "auto" and torch.cuda.is_available() else a.device
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.manual_seed(a.seed)
        if name == "xfeat":
            path = submodule("xfeat")
            sys.path.insert(0, str(path))
            from modules.xfeat import XFeat
            self.extractor = XFeat(top_k=a.max_keypoints)
            # XFeat chooses CUDA internally; move every device-dependent component.
            self.extractor.dev = torch.device(self.device)
            self.extractor.to(self.device).eval()
        else:
            path = submodule("lightglue")
            sys.path.insert(0, str(path))
            from lightglue import ALIKED, DISK, LightGlue, SuperPoint
            factory = {"superpoint": SuperPoint, "aliked": ALIKED, "disk": DISK}[name]
            self.extractor = factory(max_num_keypoints=a.max_keypoints).eval().to(self.device)
            self.matcher = LightGlue(features=name).eval().to(self.device)

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def match(self, first, second):
        torch = self.torch
        with torch.inference_mode():
            x0 = tensor_image(first, torch, self.device)
            x1 = tensor_image(second, torch, self.device)
            if self.name == "xfeat":
                f0 = self.extractor.detectAndCompute(x0[None])[0]
                f1 = self.extractor.detectAndCompute(x1[None])[0]
                n0, n1 = len(f0["keypoints"]), len(f1["keypoints"])
                if not n0 or not n1:
                    return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), n0, n1
                i, j = self.extractor.match(f0["descriptors"], f1["descriptors"])
                p0, p1 = f0["keypoints"][i], f1["keypoints"][j]
            else:
                f0 = self.extractor.extract(x0, resize=None)
                f1 = self.extractor.extract(x1, resize=None)
                n0, n1 = f0["keypoints"].shape[1], f1["keypoints"].shape[1]
                if not n0 or not n1:
                    return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), n0, n1
                matches = self.matcher({"image0": f0, "image1": f1})["matches"][0]
                p0, p1 = f0["keypoints"][0][matches[:, 0]], f1["keypoints"][0][matches[:, 1]]
            return p0.cpu().numpy(), p1.cpu().numpy(), n0, n1


def calibration(path):
    if path is None:
        return None
    data = json.loads(path.read_text())
    model = data.get("model", "pinhole")
    if model not in ("pinhole", "fisheye"):
        raise ValueError("calibration model must be pinhole or fisheye")
    K, D = np.asarray(data["K"], float), np.asarray(data["D"], float)
    size = tuple(data["image_size"])
    if K.shape != (3, 3) or len(size) != 2 or not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError("Invalid calibration: need finite K (3x3), D and image_size [width,height]")
    if model == "fisheye" and D.size != 4:
        raise ValueError("fisheye D must contain four coefficients")
    if model == "pinhole" and D.size not in (4, 5, 8, 12, 14):
        raise ValueError("pinhole D must contain 4, 5, 8, 12 or 14 coefficients")
    return model, K, D, size


def frames(a, calib):
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {a.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count, sampled, maps = 0, 0, None
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            index = count
            count += 1
            if index % a.frame_step:
                continue
            height, width = image.shape[:2]
            if calib:
                model, K, D, size = calib
                if size != (width, height):
                    raise ValueError(f"Calibration size {size} != video size {(width, height)}")
                if maps is None:
                    init = (cv2.fisheye.initUndistortRectifyMap if model == "fisheye"
                            else cv2.initUndistortRectifyMap)
                    maps = init(K, D, np.eye(3), K, size, cv2.CV_32FC1)
                image = cv2.remap(image, *maps, cv2.INTER_LINEAR)
            scale = min(1., a.max_size / max(width, height)) if a.max_size else 1.
            if scale != 1:
                image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            if min(image.shape[:2]) < 32:
                raise ValueError("Video dimensions must both be at least 32 pixels")
            yield index, index / fps if np.isfinite(fps) and fps > 0 else None, image, (width, height)
            sampled += 1
            if a.max_frames and sampled >= a.max_frames:
                break
    finally:
        cap.release()


def verify_geometry(p0, p1, threshold, seed):
    mask = np.zeros(len(p0), bool)
    if len(p0) < 8:
        return mask, "too_few_matches"
    cv2.setRNGSeed(seed)
    try:
        F, result = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, threshold, .999, 5000)
    except cv2.error:
        return mask, "fit_failed"
    if F is None or F.shape != (3, 3) or result is None or not np.isfinite(F).all():
        return mask, "fit_failed"
    return result.ravel().astype(bool), "ok"


def coverage(points, image):
    if len(points) == 0:
        return 0.
    height, width = image.shape[:2]
    cells = np.clip((points / [width, height] * [8, 6]).astype(int), [0, 0], [7, 5])
    return float(len(np.unique(cells, axis=0)) / 48)


def preview(path, first, second, p0, p1, inliers):
    canvas = np.concatenate((first, second), axis=1)
    offset = first.shape[1]
    for i in np.linspace(0, len(p0) - 1, min(len(p0), 150), dtype=int):
        start = tuple(np.round(p0[i]).astype(int))
        end = tuple(np.round(p1[i] + [offset, 0]).astype(int))
        color = (0, 220, 0) if inliers[i] else (0, 0, 220)
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write {path}")


def run_algorithm(name, a, calib, output):
    init_start = time.perf_counter()
    frontend = Classical(name, a) if name in ("klt", "orb") else Learned(name, a)
    frontend.sync()
    initialization_s = time.perf_counter() - init_start
    directory = output / name
    directory.mkdir()
    (directory / "matches").mkdir()
    (directory / "previews").mkdir()
    iterator = frames(a, calib)
    metrics, first_shape, source_size = [], None, None
    fields = ("frame0", "frame1", "time0_approx_s", "time1_approx_s", "features0", "features1",
              "matches", "inliers", "inlier_ratio", "inlier_coverage", "geometry_status",
              "frontend_ms", "geometry_ms", "warmup")
    try:
        previous = next(iterator, None)
        if previous is None:
            raise ValueError("Video has no decodable frames")
        first_shape, source_size = list(previous[2].shape[:2]), list(previous[3])
        with (directory / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for pair_number, current in enumerate(iterator):
                i0, t0, im0, _ = previous
                i1, t1, im1, _ = current
                if im0.shape != im1.shape:
                    raise ValueError("Video changed resolution during processing")
                frontend.sync()
                start = time.perf_counter()
                p0, p1, n0, n1 = frontend.match(im0, im1)
                frontend.sync()
                frontend_ms = (time.perf_counter() - start) * 1000
                valid = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
                p0, p1 = p0[valid], p1[valid]
                start = time.perf_counter()
                inliers, status = verify_geometry(p0, p1, a.ransac_threshold, a.seed)
                geometry_ms = (time.perf_counter() - start) * 1000
                row = dict(frame0=i0, frame1=i1, time0_approx_s=t0, time1_approx_s=t1,
                           features0=n0, features1=n1, matches=len(p0), inliers=int(inliers.sum()),
                           inlier_ratio=float(inliers.mean()) if len(inliers) else 0.,
                           inlier_coverage=coverage(p1[inliers], im1), geometry_status=status,
                           frontend_ms=frontend_ms, geometry_ms=geometry_ms, warmup=pair_number == 0)
                writer.writerow(row)
                handle.flush()
                metrics.append(row)
                stem = f"{i0:08d}_{i1:08d}"
                np.savez_compressed(directory / "matches" / f"{stem}.npz", points0=p0,
                                    points1=p1, inliers=inliers, frame_indices=[i0, i1])
                if pair_number < a.previews:
                    preview(directory / "previews" / f"{stem}.png", im0, im1, p0, p1, inliers)
                previous = current
                if (pair_number + 1) % 50 == 0:
                    print(f"  {name}: {pair_number + 1} pairs", flush=True)
    finally:
        iterator.close()
    if not metrics:
        raise ValueError("Need at least two sampled frames; reduce --frame-step or use a longer video")
    steady = metrics[1:]
    return dict(status="ok", pairs=len(metrics), initialization_s=initialization_s,
                processed_shape_hw=first_shape, source_size_wh=source_size,
                mean_matches=float(np.mean([r["matches"] for r in metrics])),
                mean_inliers=float(np.mean([r["inliers"] for r in metrics])),
                mean_inlier_ratio=float(np.mean([r["inlier_ratio"] for r in metrics])),
                mean_inlier_coverage=float(np.mean([r["inlier_coverage"] for r in metrics])),
                geometry_failures=sum(r["geometry_status"] != "ok" for r in metrics),
                median_frontend_ms_excluding_first=(float(np.median([r["frontend_ms"] for r in steady]))
                                                     if steady else None))


def provenance():
    try:
        result = subprocess.run(["git", "submodule", "status"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10, check=False)
        return result.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return []


def package_versions():
    versions = {}
    for name in ("torch", "torchvision", "kornia", "opencv-python", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    a = arguments()
    global cv2, np
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        print(f"Missing base dependency: {exc}. Install requirements.txt.", file=sys.stderr)
        return 2
    a.video = a.video.expanduser().resolve()
    if not a.video.is_file():
        print(f"Video does not exist: {a.video}", file=sys.stderr)
        return 2
    try:
        calib = calibration(a.calibration)
        probe = frames(a, calib)
        try:
            if next(probe, None) is None or next(probe, None) is None:
                raise ValueError("Need at least two decodable sampled frames")
        finally:
            probe.close()
    except (OSError, ValueError, KeyError, cv2.error) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (a.output or ROOT / "results" / f"{a.video.stem}_{stamp}").expanduser().resolve()
    if output.exists():
        print(f"Output already exists; choose a new directory: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True)
    np.random.seed(a.seed)
    cv2.setNumThreads(1)
    report = dict(video=str(a.video), created_utc=stamp,
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                  packages=package_versions(),
                  submodules=provenance(), calibration=json.loads(a.calibration.read_text()) if a.calibration else None,
                  notes=["Pairwise benchmark: no persistent tracks, metric pose, or ground truth accuracy.",
                         "Fundamental-matrix RANSAC inliers are diagnostic, not correct-match labels.",
                         "Times are frame index / nominal FPS, not calibrated video/IMU timestamps.",
                         "Frontend timing re-extracts both images; excludes decoding, geometry and output.",
                         "First pair is warmup; initialization includes any upstream weight downloads."],
                  algorithms={})
    if not calib:
        print("No calibration: geometry on distorted images is only a rough diagnostic.", flush=True)
    for name in a.algorithms:
        print(f"Running {name}...", flush=True)
        try:
            report["algorithms"][name] = run_algorithm(name, a, calib, output)
            print(f"  {name}: {report['algorithms'][name]['pairs']} pairs complete", flush=True)
        except Exception as exc:
            report["algorithms"][name] = dict(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"  {name}: ERROR: {exc}", file=sys.stderr, flush=True)
        finally:
            if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
                sys.modules["torch"].cuda.empty_cache()
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Results: {output}")
    return 0 if all(r["status"] == "ok" for r in report["algorithms"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
