import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 420
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

        # Map original court keypoint index (0-9) to mini court (x, y) coordinates
        self.original_to_mini_court_map = {
            3: (self.drawing_key_points[2*2], self.drawing_key_points[2*2+1]), # front-left corner -> mini court point 2
            7: (self.drawing_key_points[0*2], self.drawing_key_points[0*2+1]), # back-left corner -> mini court point 0
            8: (self.drawing_key_points[7*2], self.drawing_key_points[7*2+1]), # back-center -> mini court point 7
            2: (self.drawing_key_points[6*2], self.drawing_key_points[6*2+1]), # T-point -> mini court point 6
            0: (self.drawing_key_points[4*2], self.drawing_key_points[4*2+1]), # left wall short line -> mini court point 4
            1: (self.drawing_key_points[8*2], self.drawing_key_points[8*2+1]), # left wall service box -> mini court point 8
        }

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y

    def convert_meters_to_pixels_x(self, meters):
        return convert_meters_to_pixel_distance(
            meters,
            constants.COURT_WIDTH,
            self.court_drawing_width
        )

    def convert_meters_to_pixels_y(self, meters):
        return convert_meters_to_pixel_distance(
            meters,
            constants.COURT_LENGTH,
            self.court_drawing_height
        )

    def set_court_drawing_key_points(self):
        kp = [0] * 28  # 14 points x 2

        # Point 0 - back wall left (bottom-left)
        kp[0] = int(self.court_start_x)
        kp[1] = int(self.court_end_y)

        # Point 1 - back wall right (bottom-right)
        kp[2] = int(self.court_end_x)
        kp[3] = int(self.court_end_y)

        # Point 2 - front wall left (top-left)
        kp[4] = int(self.court_start_x)
        kp[5] = int(self.court_start_y)

        # Point 3 - front wall right (top-right)
        kp[6] = int(self.court_end_x)
        kp[7] = int(self.court_start_y)

        # Point 4 - short line left
        kp[8] = int(self.court_start_x)
        kp[9] = int(self.court_end_y - self.convert_meters_to_pixels_y(constants.SHORT_LINE_DISTANCE_FROM_BACK))

        # Point 5 - short line right
        kp[10] = int(self.court_end_x)
        kp[11] = int(kp[9])

        # Point 6 - T-point (half court line meets short line)
        kp[12] = int(self.court_start_x + self.court_drawing_width / 2)
        kp[13] = int(kp[9])

        # Point 7 - half court line bottom (back wall center)
        kp[14] = int(self.court_start_x + self.court_drawing_width / 2)
        kp[15] = int(self.court_end_y)

        # Point 8 - left service box top-left
        # sits on left wall, SERVICE_BOX_SIZE above back wall
        kp[16] = int(self.court_start_x)
        kp[17] = int(kp[9]+self.convert_meters_to_pixels_y(constants.SERVICE_BOX_SIZE))

        # Point 9 - left service box top-right (on T-line, same y as point 8)
        kp[18] = int(self.court_start_x + self.convert_meters_to_pixels_x(constants.SERVICE_BOX_SIZE))
        kp[19] = int(kp[9])

        # Point 10 - right service box top-left (on T-line, same y as point 8)
        kp[20] = int(self.court_end_x - self.convert_meters_to_pixels_x(constants.SERVICE_BOX_SIZE))
        kp[21] = int(kp[9])

        # Point 11 - right service box top-right (on right wall, same y as point 8)
        kp[22] = int(self.court_end_x)
        kp[23] = int(kp[17])

        # Point 12 - left service box bottom-right = T-line at back wall
        kp[24] = int(self.court_start_x + self.convert_meters_to_pixels_x(constants.SERVICE_BOX_SIZE))
        kp[25] = int(kp[17])

        # Point 13 - right service box bottom-left = T-line at back wall (same as 12)
        kp[26] = int(self.court_end_x - self.convert_meters_to_pixels_x(constants.SERVICE_BOX_SIZE))
        kp[27] = int(kp[17])

        self.drawing_key_points = kp

    def set_court_lines(self):
        self.lines = [
            (0, 1),    # back wall
            (2, 3),    # front wall
            (0, 2),    # left wall
            (1, 3),    # right wall
            (4, 5),    # short line (full width)
            (6, 7),    # half court T-line (T-point to back wall)
            (8,12),
            (9,12),
            (10,13),
            (11, 13),  # right service box top edge
            (8, 0),    # left service box left edge (left wall, redundant but explicit)
            (11, 1),   # right service box right edge (right wall, redundant but explicit)
        ]

    def draw_court(self, frame):
        # Draw keypoints
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            # Draw court points as dark gray (80, 80, 80) to differentiate from players (green/red)
            cv2.circle(frame, (x, y), 5, (80, 80, 80), -1)

        # Draw lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates_homography(self, point, homography):
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, homography)
        mini_court_x = transformed[0][0][0]
        mini_court_y = transformed[0][0][1]

        # Clip coordinates to keep them inside the mini-court lines
        mini_court_x = np.clip(mini_court_x, self.court_start_x, self.court_end_x)
        mini_court_y = np.clip(mini_court_y, self.court_start_y, self.court_end_y)
        
        return (mini_court_x, mini_court_y)

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        # Compute homography matrix using 4 corners of the Back-Left quadrant of the court floor
        # (since kp[2] is actually the T-point / Center Short Line floor intersection)
        kp = original_court_key_points.reshape(-1, 2)

        # 4 floor keypoints defining the Back-Left quadrant of the court floor (Camera View)
        pts_src = np.array([
            kp[0],                # Left Wall Short Line floor intersection
            kp[2],                # T-point (Center floor at short line)
            kp[7],                # Back-Left floor corner
            kp[8]                 # Back-Center floor corner
        ], dtype=np.float32)

        # Corresponding 4 coordinates in the 2D Mini-Court space
        pts_dst = np.array([
            [self.drawing_key_points[4*2], self.drawing_key_points[4*2+1]], # Left Wall Short Line
            [self.drawing_key_points[6*2], self.drawing_key_points[6*2+1]], # T-point (Center Short Line)
            [self.drawing_key_points[0*2], self.drawing_key_points[0*2+1]], # Back-Left
            [self.drawing_key_points[7*2], self.drawing_key_points[7*2+1]]  # Back-Center
        ], dtype=np.float32)

        homography, _ = cv2.findHomography(pts_src, pts_dst)

        # Compute and store homography matrix for the Front Wall
        # (kp[3] = bottom-left, kp[2] = bottom-center, kp[6] = top-left)
        front_wall_height = self.court_drawing_width * (4.57 / 6.4)
        pts_front_src = np.array([
            kp[3],                  # Bottom-Left Front Wall
            [kp[2][0], kp[3][1]],   # Bottom-Center Front Wall (approx)
            kp[6],                  # Top-Left Front Wall
            [kp[2][0], kp[6][1]]    # Top-Center Front Wall (approx)
        ], dtype=np.float32)

        pts_front_dst = np.array([
            [self.court_start_x, self.court_start_y],
            [self.court_start_x + self.court_drawing_width / 2, self.court_start_y],
            [self.court_start_x, self.court_start_y - front_wall_height],
            [self.court_start_x + self.court_drawing_width / 2, self.court_start_y - front_wall_height]
        ], dtype=np.float32)

        self.front_wall_homography, _ = cv2.findHomography(pts_front_src, pts_front_dst)

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            if frame_num >= len(ball_boxes):
                ball_dict = {}
            else:
                ball_dict = ball_boxes[frame_num]
            
            ball_box = ball_dict.get(1, [])
            ball_position = get_center_of_bbox(ball_box) if ball_box else None

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)
                mini_court_player_position = self.get_mini_court_coordinates_homography(foot_position, homography)
                output_player_bboxes_dict[player_id] = mini_court_player_position

            if ball_position:
                mini_court_ball_position = self.get_mini_court_coordinates_homography(ball_position, homography)
                output_ball_boxes.append({1: mini_court_ball_position})
            else:
                output_ball_boxes.append({})
            
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(self, frames, positions, color=None):
        for frame_num, frame in enumerate(frames):
            if frame_num < len(positions):
                for player_id, position in positions[frame_num].items():
                    x, y = position
                    x = int(x)
                    y = int(y)
                    if color is not None:
                        draw_color = color
                    else:
                        if player_id == 1:
                            draw_color = (0, 255, 0)
                        elif player_id == 2:
                            draw_color = (0, 0, 255)
                        else:
                            draw_color = (0, 255, 255)
                    cv2.circle(frame, (x, y), 5, draw_color, -1)
        return frames

    def interpolate_ball_ground_positions(self, ball_positions, player_positions, hit_log, ball_detections):
        """
        Corrects ball positions on the mini-court when the ball is high in the air
        by interpolating its ground footprint between hits and bounces.
        """
        corrected_ball_positions = [dict(x) for x in ball_positions]
        total_frames = len(ball_positions)

        # Sort the hit log by frame number
        sorted_hits = sorted(hit_log, key=lambda x: x['frame'])

        # Filter the hits to remove duplicate consecutive wall events of the same type
        filtered_hits = []
        last_event_type = None
        for event in sorted_hits:
            current_type = event['type']
            if current_type in ['front_wall', 'back_wall', 'left_wall', 'right_wall']:
                if current_type == last_event_type:
                    continue
            filtered_hits.append(event)
            last_event_type = current_type

        sorted_hits = filtered_hits

        if not sorted_hits:
            return ball_positions

        # Helper to get ground footprint coordinates for a hit/bounce event
        def get_event_ground_position(event):
            frame = event['frame']
            event_type = event['type']
            
            # 1. Racket Hit: Use player's foot position
            if event_type == 'racket':
                p_id = 1 if 'Player1' in str(event.get('player_id', '')) else 2
                if frame < len(player_positions) and p_id in player_positions[frame]:
                    return player_positions[frame][p_id]
                return ball_positions[frame].get(1)

            # 2. Bounces: Map to the physical court boundary
            raw_ball_box = ball_detections[frame].get(1, []) if frame < len(ball_detections) else []
            raw_ball_center = get_center_of_bbox(raw_ball_box) if raw_ball_box else None

            ball_pos = ball_positions[frame].get(1)
            if not ball_pos:
                return None
            
            bx, by = ball_pos
            if event_type == 'front_wall':
                if raw_ball_center and hasattr(self, 'front_wall_homography'):
                    # Warp camera coordinates via Front Wall Homography to get correct horizontal coordinate
                    pt = np.array([[[raw_ball_center[0], raw_ball_center[1]]]], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(pt, self.front_wall_homography)
                    corrected_x = transformed[0][0][0]
                    corrected_x = np.clip(corrected_x, self.court_start_x, self.court_end_x)
                    return (corrected_x, self.court_start_y)
                return (bx, self.court_start_y)
            elif event_type == 'back_wall':
                return (bx, self.court_end_y)
            elif event_type == 'left_wall':
                return (self.court_start_x, by)
            elif event_type == 'right_wall':
                return (self.court_end_x, by)
            
            # Floor bounce or unknown: Use raw projected position
            return (bx, by)

        # Interpolate between consecutive hit/bounce events
        for i in range(len(sorted_hits) - 1):
            e1 = sorted_hits[i]
            e2 = sorted_hits[i+1]
            
            f1, f2 = e1['frame'], e2['frame']
            
            pos1 = get_event_ground_position(e1)
            pos2 = get_event_ground_position(e2)
            
            if pos1 is None or pos2 is None:
                continue
                
            x1, y1 = pos1
            x2, y2 = pos2
            
            # Interpolate coordinates for all frames in between
            for f in range(f1, f2 + 1):
                if f >= total_frames:
                    break
                t = (f - f1) / (f2 - f1) if f2 > f1 else 0.0
                interp_x = x1 + t * (x2 - x1)
                interp_y = y1 + t * (y2 - y1)
                corrected_ball_positions[f] = {1: (interp_x, interp_y)}

        # Fill values before the first hit and after the last hit
        first_frame = sorted_hits[0]['frame']
        last_frame = sorted_hits[-1]['frame']
        
        first_pos = get_event_ground_position(sorted_hits[0])
        last_pos = get_event_ground_position(sorted_hits[-1])
        
        for f in range(first_frame):
            if f < total_frames and first_pos:
                if 1 in ball_positions[f]:
                    corrected_ball_positions[f] = {1: first_pos}
                    
        for f in range(last_frame + 1, total_frames):
            if f < total_frames and last_pos:
                if 1 in ball_positions[f]:
                    corrected_ball_positions[f] = {1: last_pos}

        return corrected_ball_positions

    def generate_player_heatmap(self, player_positions, player_id, output_path):
        """
        Generates a 2D density heatmap of a player's positions on the court floor,
        overlays it on the mini-court lines, and saves it as an image.
        """
        # 1. Create a blank white canvas for the court
        canvas = np.ones((self.drawing_rectangle_height, self.drawing_rectangle_width, 3), dtype=np.uint8) * 255

        # Translate drawing keypoints to local canvas coordinates
        local_keypoints = []
        for i in range(0, len(self.drawing_key_points), 2):
            local_keypoints.append(self.drawing_key_points[i] - self.start_x)
            local_keypoints.append(self.drawing_key_points[i+1] - self.start_y)

        # Draw the court lines
        for line in self.lines:
            start_point = (int(local_keypoints[line[0] * 2]), int(local_keypoints[line[0] * 2 + 1]))
            end_point = (int(local_keypoints[line[1] * 2]), int(local_keypoints[line[1] * 2 + 1]))
            cv2.line(canvas, start_point, end_point, (0, 0, 0), 2)

        # 2. Extract player coordinates local to the canvas
        pts = []
        for frame_positions in player_positions:
            if player_id in frame_positions:
                px, py = frame_positions[player_id]
                local_x = int(px - self.start_x)
                local_y = int(py - self.start_y)
                
                # Clip inside canvas
                local_x = np.clip(local_x, 0, self.drawing_rectangle_width - 1)
                local_y = np.clip(local_y, 0, self.drawing_rectangle_height - 1)
                pts.append((local_x, local_y))

        if not pts:
            print(f"No positions tracked for Player {player_id}. Cannot generate heatmap.")
            return None

        # 3. Build a density accumulation map
        density_map = np.zeros((self.drawing_rectangle_height, self.drawing_rectangle_width), dtype=np.float32)
        for pt in pts:
            # Accumulate weight at player coordinates
            cv2.circle(density_map, pt, 10, 1.0, -1)

        # Apply a Gaussian blur to get a smooth heat layout
        density_map = cv2.GaussianBlur(density_map, (25, 25), 0)

        # Normalize density values between 0 and 1
        max_val = np.max(density_map)
        if max_val > 0:
            density_map /= max_val

        # 4. Colorize the density map using JET color scheme
        density_uint8 = (density_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(density_uint8, cv2.COLORMAP_JET)

        # 5. Blend colorized heatmap with court canvas
        # Create a mask where density is active (above 2% threshold)
        mask = (density_map > 0.02)[:, :, np.newaxis]
        alpha = 0.65  # Heatmap opacity
        
        # Apply blended color where active, otherwise keep the black-and-white court canvas
        blended = np.where(mask, cv2.addWeighted(heatmap_color, alpha, canvas, 1 - alpha, 0), canvas)

        # Write to file
        cv2.imwrite(output_path, blended)
        print(f"Successfully saved Player {player_id} heatmap to {output_path}")
        return blended


