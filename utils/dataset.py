# utils/dataset.py

import os
import torch
from torch.utils.data import Dataset
from utils.video_processor import process_video, extract_audio_features

class VideoLieDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing lie detection videos.
    It assumes labels are in the filename (e.g., 'trail_lie_001.mp4').
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        
        # --- Determine Label from Filename (1 for 'lie', 0 for 'truth') ---
        filename = os.path.basename(video_path).lower()
        label = 1 if 'lie' in filename else 0

        # --- Process Video and Audio ---
        face_tensors, audio_path = process_video(video_path)
        audio_features = extract_audio_features(audio_path)
        
        # Clean up temporary audio file if it exists
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Use the first detected face tensor or a zero tensor if no face is found
        vision_data = face_tensors[0] if face_tensors else torch.zeros(3, 64, 64)
        
        # Use audio features or a zero tensor if audio processing fails
        audio_data = audio_features if audio_features is not None else torch.zeros(13)
        
        return vision_data, audio_data, torch.tensor(label, dtype=torch.long)