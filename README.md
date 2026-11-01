# Feature frontend comparison

Run the same video frame pairs through KLT, ORB, SuperPoint + LightGlue,
ALIKED + LightGlue, DISK + LightGlue, and XFeat + mutual nearest-neighbor matching.

## Setup

Python 3.10+ recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-learned.txt
git submodule update --init third_party/lightglue third_party/xfeat
```

## Run

```bash
python run_features.py /path/to/recording.MP4
```


For classical methods alone, install just `requirements.txt`; no submodules or
PyTorch are needed. Missing/uninitialized learned methods are reported as errors,
other methods still run, and the process exits with status 1 for any failure.
Invalid input exits with status 2. Existing output directories are never overwritten.
`--help` works without installing dependencies.

## Results

- `summary.json`: settings, versions, submodule revisions, calibration, per-method
  status, mean match/inlier counts, coverage, geometry failures, and median runtime.
- `<algorithm>/metrics.csv`: per-pair counts, nominal times, timing, and diagnostics.
- `<algorithm>/matches/*.npz`: corresponding `points0`, `points1` (N x 2), boolean
  `inliers`, and original video frame indices. Coordinates are in the processed
  (optionally undistorted, then resized) images.
- `<algorithm>/previews/*.png`: side-by-side matches, green RANSAC inliers and red
  rejected matches; at most 150 uniformly sampled matches are drawn per preview.


## Optional GoPro calibration

Without calibration, geometric verification on distorted GoPro images is only
diagnostic. Disable electronic stabilization for ordinary fixed-camera geometry.
Use calibration for the EXACT recording mode and source resolution. Supply a JSON
file containing these keys (values must come from your calibration):

```text
model: "pinhole" or "fisheye"
K: 3x3 camera intrinsic matrix
D: OpenCV distortion coefficients (4 for fisheye)
image_size: [source_width, source_height]
```

```bash
python run_features.py recording.MP4 --calibration calibration.json
```

Frames are undistorted with the original K, then resized identically for all methods.
For downstream geometry, scale K's first/second rows by processed width/source
width and processed height/source height, respectively. Nominal timestamps here
are frame index divided by FPS, **not** exposure timestamps or synchronized IMU time;
do not use them as a calibrated VIO clock without further work.

## Upstreams

| Submodule | Frontends |
| --- | --- |
| [LightGlue](https://github.com/cvg/LightGlue) | SuperPoint, ALIKED, DISK + their matched LightGlue weights |
| [XFeat](https://github.com/verlab/accelerated_features) | XFeat + mutual nearest-neighbor matching |
| [OpenCV](https://github.com/opencv/opencv) | KLT and ORB source reference; runtime uses Python wheel |
