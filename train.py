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
        patience=50,              # Early stopping patience
        amp=True,               # Enable Automatic Mixed Precision (AMP)
        save=True,              # Save the model during training
        save_period=-1,         # Save every epoch (or you can set a custom period)
        plots=True,             # Plot training progress
        verbose=True            # Display verbose output
    )

if __name__ == "__main__":
    model_name = 'yolov8s-seg.pt'  # 'yolov5s.pt' or 'yolov8s-seg.pt'
    data_path = 'synthetic_dataset_segmentation/data.yaml'

    train_yolo(model_name, data_path)
