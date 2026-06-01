from ultralytics import YOLO 

model = YOLO('models/best.pt')

model.predict('input_videos/input_video1.mp4',conf=0.2,save=True)