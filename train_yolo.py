from ultralytics import YOLO
import os

def train_table_detector():
    """Train a custom YOLO model for table occupancy detection"""
    
    # Load a pretrained model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='dataset.yaml',  # Path to your dataset YAML
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.01,
        device=0,  # 0 for GPU, None for CPU
        workers=8,
        save=True,
        pretrained=True
    )
    
    print("Training completed!")
    return model

if __name__ == '__main__':
    train_table_detector()