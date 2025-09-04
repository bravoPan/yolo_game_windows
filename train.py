from ultralytics import YOLO
import os
import torch

# import torch
# print(torch.cuda.is_available())

# # Set working directory to dataset folder
# Check current directory
def main():
    print(f"Current working directory: {os.getcwd()}")

    # Set working directory to dataset folder (use relative path)
    dataset_path = os.path.join(os.path.dirname(__file__), "annotated_ds")
    # print(f"Dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        # print(f"Error: Dataset folder not found at {dataset_path}")
        # print("Available folders:", [d for d in os.listdir(".") if os.path.isdir(d)])
        exit(1)

    os.chdir(dataset_path)
    # print(f"Changed to directory: {os.getcwd()}")


    # Load a pre-trained YOLO model
    model = YOLO('yolov8n.pt')  # Load YOLOv8 nano model

    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=300,           # Number of epochs
        imgsz=1440,           # Image size
        batch=16,            # Batch size
        device=0,       # Use GPU if available, else CPU
        patience=20,         # Early stopping patience
        save=True,           # Save checkpoints
        project='runs/train', # Project name
        name='yolo_game_model' # Experiment name
    )

    # Save the trained model
    model.save('yolo_game_model.pt')

    # print("Training completed!")
    # print(f"Best model saved as: best_game_model.pt")

if __name__ == '__main__':
    main()