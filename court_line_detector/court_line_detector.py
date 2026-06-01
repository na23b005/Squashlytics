import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision import models


class CourtLineDetector:

    def __init__(self, model_path):

        self.model = models.resnet50(weights=None)

        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            20      # 10 keypoints × 2
        )

        checkpoint = torch.load(
            model_path,
            map_location="cpu"
        )

        # Handles both:
        # torch.save(model.state_dict())
        # and checkpoint dictionaries
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image):
        original_h, original_w = image.shape[:2]
        
        image_square = cv2.resize(image, (640, 640))
        image_rgb = cv2.cvtColor(image_square, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()

        keypoints[::2]  *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        # Manual offsets: {keypoint_index: (x_offset, y_offset)}
        # Negative x = move left, Negative y = move up
        # offsets = {
        #     0: (0,0),   # move left
        #     1: (0,0),   # move left
        #     # 2: (  0, -50),   # move up
        #     3: (0,10),
        #     4:(0,10),
        #     5:(0,10),
        #     7: (-25,   0),   # move left
        #     # 9: (  25, -75),   # move up
        # }

        # for kp_idx, (dx, dy) in offsets.items():
        #     keypoints[kp_idx * 2]     += dx   # x
        #     keypoints[kp_idx * 2 + 1] += dy   # y

        return keypoints

    def draw_keypoints(self, image, keypoints):

        for i in range(0, len(keypoints), 2):

            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            cv2.circle(
                image,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                image,
                str(i // 2),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return image

    def draw_keypoints_on_video(
        self,
        video_frames,
        keypoints
    ):

        output_frames = []

        for frame in video_frames:

            frame = self.draw_keypoints(
                frame.copy(),
                keypoints
            )

            output_frames.append(frame)

        return output_frames