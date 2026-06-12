# main.py

import argparse
from agents.lie_detect_agent import LieDetectAgent

def analyze_video(video_path, statement):
    """
    Initializes the agent and runs analysis on a single video file.
    """
    print("--- Initializing Lie Detection Agent ---")
    agent = LieDetectAgent()
    print("Agent initialized. Starting analysis...")
    
    decision, score, reasoning = agent.analyze_from_video(
        video_path=video_path,
        text_input=statement
    )
    
    # --- Print the Report ---
    print("\n--- 🕵️ Analysis Report 🕵️ ---")
    print(f"Video File: '{video_path}'")
    print("-" * 25)
    print("Reasoning Trace:")
    for i, step in enumerate(reasoning, 1):
        print(f"  {i}. {step}")
    print("-" * 25)
    print(f"Final Decision: {decision.upper()}")
    print(f"Lie Confidence Score: {score:.2f}")
    print("------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal Lie Detection Analysis")
    parser.add_argument("video", type=str, help="Full path to the video file to be analyzed.")
    parser.add_argument("--text", type=str, default=None, help="Optional text statement to analyze with the video.")
    
    args = parser.parse_args()
    
    analyze_video(args.video, args.text)