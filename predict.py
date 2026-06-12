import torch
import argparse
from models.vision_model import VisionCNN
from utils.video_processor import process_video
import torch.nn as nn

# --- 1. Configuration ---
MODEL_PATH = 'vision_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Define Prediction Function ---
def predict(video_path):
    """
    Loads the trained model and makes a prediction on a single video.
    It handles multiple video chunks by taking a majority vote on predictions.
    """
    model = VisionCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print(f"Model loaded from {MODEL_PATH}")
    
    print(f"Processing video: {video_path}...")
    try:
        # --- THIS IS THE FINAL FIX ---
        # The function returns a list of video tensors and the audio path.
        vision_tensor_list, audio_path = process_video(video_path)
        
        # Check if the processing returned a list.
        if not isinstance(vision_tensor_list, list):
            print(f"Video processing failed. Expected a list of tensors but got {type(vision_tensor_list)}")
            return
            
    except Exception as e:
        print(f"An unexpected error occurred during video processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Make a Prediction for Each Chunk ---
    predictions = []
    class_labels = ['Truth', 'Lie']
    
    with torch.no_grad():
        for i, vision_tensor in enumerate(vision_tensor_list):
            # Add the batch dimension and move to device
            vision_data = vision_tensor.unsqueeze(0).to(DEVICE)
            
            output = model(vision_data)
            _, predicted_idx = torch.max(output.data, 1)
            prediction = class_labels[predicted_idx.item()]
            predictions.append(prediction)
            print(f"  - Prediction for video chunk {i+1}: {prediction}")

    # --- Aggregate Results with a Majority Vote ---
    if not predictions:
        print("Could not make any predictions.")
        return
        
    final_prediction = max(set(predictions), key=predictions.count)
    
    print("-" * 30)
    print(f"Final Decision (Majority Vote): {final_prediction}")
    print("-" * 30)

# --- 3. Run the script from the command line ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Lie or Truth from a video file.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file for prediction.')
    
    args = parser.parse_args()
    
    predict(args.video_path)