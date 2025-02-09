from ultralytics import YOLO

def train_yolo(model_name, data_path):
    model = YOLO(model_name)  # Load the pre-trained YOLOv5 model

    # Training the model with custom parameters
    model.train(
        data=data_path,        # Path to your data.yaml
        epochs=100,              # Set number of epochs
        batch=16,           # Set batch size
        imgsz=640,               # Image size
        device='cuda',           # Use GPU
        workers=4,               # Number of data loading workers
        optimizer='AdamW',         # Optimizer (SGD or Adam)
        lr0=0.01,                # Learning rate
        patience=50              # Early stopping patience
    )

if __name__ == "__main__":
    model_name = 'yolov5s.pt'  # 'yolov5s.pt' or 'yolov8s-seg.pt'
    data_path = "/sise/home/oryanyeh/muze.ai/synthetic_dataset/data.yaml"

    train_yolo(model_name, data_path)
