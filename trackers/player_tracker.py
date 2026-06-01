from ultralytics import YOLO
import cv2
import pickle


class PlayerTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml"
        )[0]

        player_dict = {}

        best_player1 = None
        best_player1_conf = -1

        best_player2 = None
        best_player2_conf = -1

        for box in results.boxes:

            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            bbox = box.xyxy.tolist()[0]

            # Player1
            if cls_id == 0:
                if conf > best_player1_conf:
                    best_player1_conf = conf
                    best_player1 = bbox

            # Player2
            elif cls_id == 1:
                if conf > best_player2_conf:
                    best_player2_conf = conf
                    best_player2 = bbox

        if best_player1 is not None:
            player_dict[1] = best_player1

        if best_player2 is not None:
            player_dict[2] = best_player2

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):

        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):

            for player_id, bbox in player_dict.items():

                x1, y1, x2, y2 = bbox

                if player_id == 1:
                    color = (0, 255, 0)
                    label = "Player1"
                else:
                    color = (0, 0, 255)
                    label = "Player2"

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            output_video_frames.append(frame)

        return output_video_frames