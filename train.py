from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

model.train(data='data.yaml', epochs=1, imgsz=640)
#model.train(data='/content/gdrive/MyDrive/Food-Detect/apple/config.yaml', epochs=10, imgsz=640)