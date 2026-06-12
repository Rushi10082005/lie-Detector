# agents/lie_detect_agent.py

import torch
import os
from models.vision_model import VisionCNN
from models.audio_model import AudioLSTM
from models.text_model import text_model_predict
from utils.video_processor import process_video, extract_audio_features

class LieDetectAgent:
    """
    The agent that loads trained models and analyzes multi-modal inputs.
    """
    def __init__(self):
        # --- Initialize and Load the Vision Model ---
        self.vision_model = VisionCNN()
        model_path = 'vision_model.pth'
        try:
            self.vision_model.load_state_dict(torch.load(model_path))
            print(f"Loaded trained vision model weights from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Trained model file not found at {model_path}. Vision model is UNTRAINED.")
        self.vision_model.eval() # Set to evaluation mode

        # --- Initialize Audio and Text Models ---
        self.audio_model = AudioLSTM()
        self.audio_model.eval()
        # The text model is a global instance loaded by its module.

    def analyze_from_video(self, video_path, text_input=None):
        """
        Analyzes a single video file by processing its visual and audio components.
        """
        face_tensors, audio_path = process_video(video_path)
        
        vision_prob = 0.5
        if face_tensors:
            with torch.no_grad():
                batch = torch.stack(face_tensors)
                outputs = self.vision_model(batch)
                probabilities = torch.softmax(outputs, dim=1)
                vision_prob = probabilities[:, 1].mean().item()
        
        audio_features = extract_audio_features(audio_path)
        audio_prob = 0.5
        if audio_features is not None:
             with torch.no_grad():
                outputs = self.audio_model(audio_features.unsqueeze(0).unsqueeze(0))
                probabilities = torch.softmax(outputs, dim=1)
                audio_prob = probabilities[0, 1].item()
        
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            
        text_prob = 0.5
        if text_input:
            probs = text_model_predict([text_input])
            text_prob = float(probs[0][1])

        return self._generate_report(vision_prob, audio_prob, text_prob, text_input)

    def _generate_report(self, vision_prob, audio_prob, text_prob, text_input):
        """Helper function to create the final reasoning trace and decision."""
        reasoning_trace = []
        modality_results = [{'lie': vision_prob}, {'lie': audio_prob}]
        
        reasoning_trace.append(f"Vision analysis suggests lie probability {vision_prob:.2f}.")
        reasoning_trace.append(f"Audio analysis suggests lie probability {audio_prob:.2f}.")
        
        if text_input:
            reasoning_trace.append(f"Text analysis for '{text_input}' suggests lie probability {text_prob:.2f}.")
            modality_results.append({'lie': text_prob})

        total_score = sum(res['lie'] for res in modality_results) / len(modality_results)
        decision = 'lie' if total_score >= 0.5 else 'truth'
        
        reasoning_trace.append(f"Final combined decision is '{decision.upper()}' with a lie confidence of {total_score:.2f}.")
        
        return decision, total_score, reasoning_trace