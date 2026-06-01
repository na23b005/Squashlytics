import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters)

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
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

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