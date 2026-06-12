# utils/data_loader.py

import os
import glob

def load_video_paths(root_data_path):
    """
    Scans the train and test directories to find all video files.
    
    Args:
        root_data_path (str): The path to the main data folder 
                               (e.g., 'data/real lie detection').
    
    Returns:
        A dictionary containing lists of file paths for train and test sets.
    """
    if not os.path.exists(root_data_path):
        print(f"Error: Data path not found at '{root_data_path}'")
        return None

    train_path = os.path.join(root_data_path, 'train')
    test_path = os.path.join(root_data_path, 'test')

    # Use glob to find all files with common video extensions
    train_files = glob.glob(os.path.join(train_path, '**', '*.mp4'), recursive=True)
    train_files.extend(glob.glob(os.path.join(train_path, '**', '*.avi'), recursive=True))

    test_files = glob.glob(os.path.join(test_path, '**', '*.mp4'), recursive=True)
    test_files.extend(glob.glob(os.path.join(test_path, '**', '*.avi'), recursive=True))
    
    print(f"Found {len(train_files)} training videos and {len(test_files)} testing videos.")

    return {
        "train": train_files,
        "test": test_files
    }

# Example of how to use this function:
if __name__ == '__main__':
    # This part runs only if you execute this file directly for testing
    data_path = '../data/real lie detection' # Assumes running from inside the utils folder
    video_files = load_video_paths(data_path)
    if video_files:
        print("\nFirst 5 training files:")
        print(video_files['train'][:5])