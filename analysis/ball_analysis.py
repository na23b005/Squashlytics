import pickle
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from collections import deque, Counter
from shapely.geometry import Point, Polygon


# ─────────────────────────────────────────
# COURT LINE DETECTOR
# ─────────────────────────────────────────
class CourtLineDetector:

    def __init__(self, model_path):
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 20)

        checkpoint = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        original_h, original_w = image.shape[:2]
        image_square = cv2.resize(image, (640, 640))
        image_rgb    = cv2.cvtColor(image_square, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        keypoints[::2]  *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        return keypoints


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_ball_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_in_polygon(point, polygon):
    try:
        return Polygon(polygon).contains(Point(point))
    except Exception:
        return False


# ─────────────────────────────────────────
# COURT ZONE BUILDER
# ─────────────────────────────────────────
def build_court_zones(keypoints):
    kp = keypoints.reshape(-1, 2)  # shape (10, 2)

    court_half_w = kp[2][0] - kp[0][0]
    center = kp[2]

    def right_of(kp_left):
        dx = center[0] - kp_left[0]
        return np.array([center[0] + dx, kp_left[1]])

    r0 = np.array([center[0] + court_half_w, kp[0][1]])  # noqa: F841
    r1 = right_of(kp[1])   # noqa: F841
    r3 = right_of(kp[3])
    r6 = right_of(kp[6])
    r7 = right_of(kp[7])
    r9 = right_of(kp[9])

    WALL_BUFFER = 40

    zones = {
        'front_wall': np.array([
            kp[6] + [-WALL_BUFFER, -WALL_BUFFER],
            r6    + [ WALL_BUFFER, -WALL_BUFFER],
            r3    + [ WALL_BUFFER,  WALL_BUFFER],
            kp[3] + [-WALL_BUFFER,  WALL_BUFFER],
        ]),
        'left_wall': np.array([
            kp[6] + [-WALL_BUFFER,  0],
            kp[3] + [-WALL_BUFFER,  0],
            kp[7] + [-WALL_BUFFER,  WALL_BUFFER],
            kp[9] + [-WALL_BUFFER, -WALL_BUFFER],
        ]),
        'right_wall': np.array([
            r6 + [ WALL_BUFFER,  0],
            r3 + [ WALL_BUFFER,  0],
            r7 + [ WALL_BUFFER,  WALL_BUFFER],
            r9 + [ WALL_BUFFER, -WALL_BUFFER],
        ]),
        'back_wall': np.array([
            kp[9] + [0, -WALL_BUFFER],
            kp[7] + [0,  WALL_BUFFER],
            kp[8] + [0,  WALL_BUFFER],
            r7    + [0,  WALL_BUFFER],
            r9    + [0, -WALL_BUFFER],
        ]),
        'floor': np.array([kp[3], r3, r7, kp[7]]),
    }
    return zones


# ─────────────────────────────────────────
# BALL TRACKER
# ─────────────────────────────────────────
class BallTracker:
    def __init__(self, buffer_size=8, cooldown=8):
        self.history        = deque(maxlen=buffer_size)
        self.last_hit_frame = -999
        self.COOLDOWN       = cooldown

    def update(self, bbox, frame_id):
        cx, cy = get_ball_center(bbox)
        self.history.append((cx, cy, frame_id))

    def get_velocity_vectors(self):
        if len(self.history) < 3:
            return None
        pts = list(self.history)
        v1 = np.array([pts[-2][0] - pts[-3][0], pts[-2][1] - pts[-3][1]], dtype=float)
        v2 = np.array([pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]], dtype=float)
        return v1, v2

    def detect_bounce(self, threshold=25):
        vels = self.get_velocity_vectors()
        if vels is None:
            return False, 0.0
        v1, v2 = vels
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm < 1e-6:
            return False, 0.0
        cos_angle = np.dot(v1, v2) / norm
        angle     = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return angle > threshold, angle


# ─────────────────────────────────────────
# VELOCITY REFINEMENT
# ─────────────────────────────────────────
def refine_with_velocity(v1, v2, fallback_type):
    y_flip = (v1[1] > 0 and v2[1] < 0)
    x_flip = (v1[0] * v2[0] < 0)

    if y_flip and not x_flip:
        return 'floor'
    if x_flip and not y_flip:
        return 'front_wall' if v1[0] > 0 else 'back_wall'
    if y_flip and x_flip:
        return 'corner'
    return fallback_type


# ─────────────────────────────────────────
# HIT CLASSIFIER
# ─────────────────────────────────────────
def classify_hit(ball_bbox, player_bboxes, keypoints, ball_tracker, frame_id):
    ball_tracker.update(ball_bbox, frame_id)

    bounced, angle = ball_tracker.detect_bounce(threshold=25)
    if not bounced:
        return None

    # ── Cooldown: skip if too soon after last hit ──
    if frame_id - ball_tracker.last_hit_frame < ball_tracker.COOLDOWN:
        return None
    ball_tracker.last_hit_frame = frame_id

    ball_cx, ball_cy = get_ball_center(ball_bbox)
    ball_point = (ball_cx, ball_cy)

    # ── Check 1: Racket Hit ──
    for pid, pbbox in player_bboxes.items():
        px1, py1, px2, py2 = pbbox
        racket_zone = (px1 - 80, py1 - 120, px2 + 80, py2)

        if (racket_zone[0] < ball_cx < racket_zone[2] and
                racket_zone[1] < ball_cy < racket_zone[3]):
            return {
                'frame':     frame_id,
                'type':      'racket',
                'player_id': f"Player{pid}",
                'angle':     round(angle, 2),
                'position':  (round(ball_cx), round(ball_cy))
            }

    # ── Check 2: Zone-based wall / floor ──
    court_zones = build_court_zones(keypoints)
    for zone_name, polygon in court_zones.items():
        if point_in_polygon(ball_point, polygon):
            vels    = ball_tracker.get_velocity_vectors()
            refined = refine_with_velocity(vels[0], vels[1], zone_name) if vels else zone_name
            return {
                'frame':     frame_id,
                'type':      refined,
                'player_id': None,
                'angle':     round(angle, 2),
                'position':  (round(ball_cx), round(ball_cy))
            }

    return {
        'frame':     frame_id,
        'type':      'unknown',
        'player_id': None,
        'angle':     round(angle, 2),
        'position':  (round(ball_cx), round(ball_cy))
    }


# ─────────────────────────────────────────
# HIGH-LEVEL API (used by main.py)
# ─────────────────────────────────────────
def process_all_hits(ball_detections, player_detections, court_keypoints):
    """
    Run hit detection across all frames.
    Returns hit_log: list of dicts {frame, type, player_id, angle, position}.
    Also prints a formatted table to stdout.
    """
    tracker      = BallTracker(buffer_size=8, cooldown=8)
    hit_log      = []
    total_frames = max(len(ball_detections), len(player_detections))

    print("=" * 68)
    print(f"{'FRAME':<8} {'HIT TYPE':<15} {'PLAYER':<12} {'ANGLE':>7}  {'POSITION'}")
    print("=" * 68)

    for frame_id in range(total_frames):
        ball_frame   = ball_detections[frame_id]   if frame_id < len(ball_detections)   else {}
        player_frame = player_detections[frame_id] if frame_id < len(player_detections) else {}

        if not ball_frame:
            continue

        ball_bbox = list(ball_frame.values())[0]
        result    = classify_hit(ball_bbox, player_frame, court_keypoints, tracker, frame_id)

        if result:
            hit_log.append(result)
            player_str = result['player_id'] if result['player_id'] else "—"
            print(
                f"{result['frame']:<8} "
                f"{result['type']:<15} "
                f"{player_str:<12} "
                f"{result['angle']:>7.1f}°  "
                f"{result['position']}"
            )

    print("=" * 68)

    # Summary
    print("\n─── Hit Summary ───")
    type_counts = Counter(h['type'] for h in hit_log)
    for hit_type, count in sorted(type_counts.items()):
        print(f"  {hit_type:<15}: {count} hits")
    print(f"  {'TOTAL':<15}: {len(hit_log)} hits")

    return hit_log


def build_frame_counts(hit_log, total_frames):
    """
    Build per-frame running hit counts for Player1 and Player2 only.
    Returns dict[frame_id] → {'player1': int, 'player2': int}
    """
    p1 = 0
    p2 = 0
    counts    = {}
    hit_index = 0

    for frame_id in range(total_frames):
        if hit_index < len(hit_log) and hit_log[hit_index]['frame'] == frame_id:
            h = hit_log[hit_index]
            if h['type'] == 'racket':
                if h['player_id'] == 'Player1':
                    p1 += 1
                elif h['player_id'] == 'Player2':
                    p2 += 1
            hit_index += 1

        counts[frame_id] = {'player1': p1, 'player2': p2}

    return counts