from minicourt import MiniCourt
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import sys
sys.path.append("analysis")
from ball_analysis import process_all_hits, build_frame_counts

import cv2


def main():
    # ─────────────────────────────────────────
    # Read Video
    # ─────────────────────────────────────────
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # ─────────────────────────────────────────
    # Trackers
    # ─────────────────────────────────────────
    player_tracker = PlayerTracker(model_path='models/best_player.pt')
    ball_tracker   = BallTracker(model_path='models/best_ball.pt')

    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections.pkl"
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/ball_detections.pkl"
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # ─────────────────────────────────────────
    # Court Keypoints
    # ─────────────────────────────────────────
    court_model_path    = "models/keypoints_best_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints     = court_line_detector.predict(video_frames[0])
    print("Court Keypoints:", court_keypoints)

    # ─────────────────────────────────────────
    # Mini Court
    # ─────────────────────────────────────────
    mini_court = MiniCourt(video_frames[0])

    # ─────────────────────────────────────────
    # Hit Detection 
    # ─────────────────────────────────────────
    hit_log      = process_all_hits(ball_detections, player_detections, court_keypoints)
    frame_counts = build_frame_counts(hit_log, len(video_frames))

    # ─────────────────────────────────────────
    # Draw Pipeline
    # ─────────────────────────────────────────
    mini_court_player_positions, mini_court_ball_positions = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections,
        ball_detections,
        court_keypoints
    )

    # Correct ball projections for when the ball is high in the air
    mini_court_ball_positions = mini_court.interpolate_ball_ground_positions(
        mini_court_ball_positions,
        mini_court_player_positions,
        hit_log,
        ball_detections
    )


    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, mini_court_player_positions)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, mini_court_ball_positions, color=(0, 255, 255))

    # ─────────────────────────────────────────
    # Draw Scoreboard on Each Frame
    # ─────────────────────────────────────────
    for i, frame in enumerate(output_video_frames):
        counts = frame_counts.get(i, {'player1': 0, 'player2': 0})

        # Semi-transparent background box
        box_x1, box_y1 = 10, 50
        box_x2, box_y2 = 290, 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Player 1
        cv2.putText(
            frame,
            f"Player 1 Hits : {counts['player1']}",
            (box_x1 + 10, box_y1 + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
        )

        # Player 2
        cv2.putText(
            frame,
            f"Player 2 Hits : {counts['player2']}",
            (box_x1 + 10, box_y1 + 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
        )

        # Frame number
        cv2.putText(
            frame,
            f"Frame: {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    # ─────────────────────────────────────────
    # Generate Heatmaps
    # ─────────────────────────────────────────
    mini_court.generate_player_heatmap(mini_court_player_positions, 1, "output_videos/player_1_heatmap.png")
    mini_court.generate_player_heatmap(mini_court_player_positions, 2, "output_videos/player_2_heatmap.png")

    # ─────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()