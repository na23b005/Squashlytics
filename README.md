# Squash Analysis

A computer vision pipeline that tracks players and the ball in a squash match video, detects hits, and overlays a mini-court visualization with scoreboard stats.

## Features

- **Player & Ball Tracking** — YOLOv8/YOLOv11-based detection with interpolation for missed ball frames
- **Court Keypoint Detection** — Custom CNN that localizes court lines in the first frame
- **Mini-Court Projection** — Maps real-world positions onto a top-down court diagram
- **Hit Detection** — Assigns each ball hit to the nearest player
- **Scoreboard Overlay** — Displays per-player hit counts on every output frame
- **Heatmaps** — Generates positional heatmaps for each player
- **Dashboard** — Interactive web dashboard to visualize match stats, heatmaps, and hit counts.

## Project Structure

```
squash_analysis/
├── main.py                  # Entry point
├── trackers/                # Player & ball tracker classes
├── court_line_detector/     # Court keypoint model & inference
├── minicourt/               # Mini-court drawing & coordinate projection
├── analysis/                # Hit detection logic
├── utils/                   # Video I/O helpers
├── constants/               # Shared constants
├── training/                # Training notebooks (Jupyter)
├── input_videos/            # Input video (not tracked in git)
├── output_videos/           # Output video & heatmaps (not tracked in git)
├── models/                  # Model weights (not tracked in git)
└── tracker_stubs/           # Cached detection pickles (not tracked in git)
```

## Requirements

```bash
pip install ultralytics opencv-python torch torchvision numpy
```

## Usage

1. Place your input video at `input_videos/input_video.mp4`
2. Download model weights into `models/`:
   - `best_player.pt`
   - `best_ball.pt`
   - `keypoints_best_model.pth`
3. Run:

```bash
python main.py
```

Output video is saved to `output_videos/output_video.avi`. Heatmaps are saved as `player_1_heatmap.png` and `player_2_heatmap.png` in the same folder.

## Notes

- Model weights, videos, and detection stubs are **excluded from git** due to file size.
- Set `read_from_stub=False` in `main.py` if you want fresh detections instead of cached ones.

## Dashboard

A web-based dashboard is included for interactive post-match analysis. It displays:

- Per-player hit counts and timelines
- Player positional heatmaps
- Ball trajectory overlaid on the mini-court

Run the dashboard after processing a video to explore the match data visually.
