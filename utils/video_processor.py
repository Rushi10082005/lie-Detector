# utils/video_processor.py

import cv2
import torch
import numpy as np
from moviepy.editor import VideoFileClip
import os
import librosa

# --- Configuration ---
# Make sure 'haarcascade_frontalface_default.xml' is in your project's root folder
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
IMAGE_SIZE = (64, 64)  # The size our VisionCNN model expects

def process_video(video_path):
    """
    Processes a single video file to extract facial frames and an audio path.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None, None

    # --- 1. Extract and Process Facial Frames ---
    cap = cv2.VideoCapture(video_path)
    face_tensors = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_crop, IMAGE_SIZE)
            tensor = torch.from_numpy(resized_face).permute(2, 0, 1).float() / 255.0
            face_tensors.append(tensor)
            break 
    
    cap.release()

    # --- 2. Extract Audio ---
    audio_path = None
    try:
        video_clip = VideoFileClip(video_path)
        audio_filename = os.path.basename(video_path) + ".wav"
        temp_audio_path = os.path.join("temp_audio", audio_filename)
        os.makedirs("temp_audio", exist_ok=True)
        video_clip.audio.write_audiofile(temp_audio_path, logger=None)
        audio_path = temp_audio_path
    except Exception as e:
        audio_path = None

    if not face_tensors:
        return None, audio_path

    return face_tensors, audio_path

def extract_audio_features(audio_path, n_mfcc=13):
    """
    Extracts MFCC features from an audio file.
    """
    if not audio_path or not os.path.exists(audio_path):
        return None
        
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        return torch.from_numpy(mfccs_mean).float()
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None