# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_loader import load_video_paths
from utils.dataset import VideoLieDataset
from models.vision_model import VisionCNN

# --- 1. Hyperparameters & Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCHS = 10
DATA_PATH = 'data/Real-life Deception'
MODEL_SAVE_PATH = 'vision_model.pth'

# --- 2. Load Data ---
print("Loading video file paths...")
video_paths = load_video_paths(DATA_PATH)
if not video_paths:
    exit("Could not load video paths. Please check the DATA_PATH.")

train_dataset = VideoLieDataset(video_paths['train'])
test_dataset = VideoLieDataset(video_paths['test'])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Data loading complete.")

# --- 3. Initialize Model, Loss, and Optimizer ---
model = VisionCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("Model, Loss, and Optimizer initialized.")

# --- 4. Training Loop ---
print("\n--- Starting Training ---")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    for i, (vision_data, _, labels) in enumerate(train_loader):
        vision_data, labels = vision_data.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(vision_data)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

    # --- 5. Validation Loop ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for vision_data, _, labels in test_loader:
            vision_data, labels = vision_data.to(DEVICE), labels.to(DEVICE)
            outputs = model(vision_data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f} %")

print("--- Training Finished ---")

# --- 6. Save the Trained Model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")