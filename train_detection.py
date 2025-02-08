from ultralytics import YOLO

def train_yolov5():
    model = YOLO('yolov5s.pt')  # Load the pre-trained YOLOv5 model

    # Training the model with custom parameters
    model.train(
        data="/sise/home/oryanyeh/muze.ai/synthetic_dataset/data.yaml",        # Path to your data.yaml
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
    train_yolov5()
