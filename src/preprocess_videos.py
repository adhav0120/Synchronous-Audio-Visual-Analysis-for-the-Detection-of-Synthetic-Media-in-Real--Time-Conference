# src/preprocess_videos.py

import cv2
from mtcnn import MTCNN
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

# --- Configuration ---
# Define the root of your project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the FaceForensics++ dataset
VIDEO_DATA_ROOT = PROJECT_ROOT / "FaceForensics++_C23"

# Path to the output directory for extracted faces
OUTPUT_FACE_DATA_ROOT = PROJECT_ROOT / "face_data"

# --- Main Script ---

def extract_faces(video_path, output_dir, detector, batch_size=32):
    """
    Extracts faces from a single video file using batch processing and saves them as images.

    Args:
        video_path (Path): The path to the input video file.
        output_dir (Path): The directory to save the extracted face images.
        detector (MTCNN): The pre-initialized MTCNN face detector.
        batch_size (int): The number of frames to process in a single batch.
    """
    # Create subdirectories for real and fake faces
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        frames_batch = []
        original_frames_batch = [] # Store original frames for cropping
        frame_indices = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video

            # Process every Nth frame to speed things up (optional)
            if frame_count % 15 == 0: 
                # Resize frames to a consistent size before batching
                frame_resized = cv2.resize(frame, (256, 256))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                frames_batch.append(frame_rgb)
                original_frames_batch.append(frame) # Keep original for high-res cropping
                frame_indices.append(frame_count)

                # When the batch is full, process it
                if len(frames_batch) == batch_size:
                    process_batch(frames_batch, original_frames_batch, frame_indices, video_path, output_dir, detector)
                    # Reset the batch
                    frames_batch = []
                    original_frames_batch = []
                    frame_indices = []

            frame_count += 1
        
        # Process any remaining frames in the last batch
        if frames_batch:
            process_batch(frames_batch, original_frames_batch, frame_indices, video_path, output_dir, detector)
            
        cap.release()

    except Exception as e:
        print(f"An error occurred while processing {video_path}: {e}")

def process_batch(frames_batch, original_frames, frame_indices, video_path, output_dir, detector):
    """Detects and saves faces from a batch of frames."""
    # --- FIX: Pass the list of frames directly to the detector ---
    # The MTCNN library can handle a list of images, which is more robust than a single numpy array.
    results = detector.detect_faces(frames_batch)
    
    video_name = video_path.stem
    
    for i, result_list in enumerate(results):
        # The result from detect_faces is a list of dictionaries.
        if result_list: # Check if any face was detected in this frame
            original_frame = original_frames[i]
            frame_idx = frame_indices[i]

            # Get the first and most confident face
            box = result_list[0]['box']
            x1, y1, width, height = box
            
            # Scale coordinates back to the original frame size
            orig_h, orig_w, _ = original_frame.shape
            resized_h, resized_w, _ = frames_batch[i].shape
            
            x1 = int(x1 * (orig_w / resized_w))
            y1 = int(y1 * (orig_h / resized_h))
            width = int(width * (orig_w / resized_w))
            height = int(height * (orig_h / resized_h))
            
            x2, y2 = x1 + width, y1 + height
            
            # Ensure coordinates are within the frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_frame.shape[1], x2), min(original_frame.shape[0], y2)

            # Crop the face from the original, high-resolution frame
            face = original_frame[y1:y2, x1:x2]
            
            if face.size > 0:
                # Construct a unique filename
                face_filename = f"{video_name}_frame{frame_idx}.png"
                save_path = output_dir / face_filename
                
                # Save the face image
                cv2.imwrite(str(save_path), face)


def main():
    """
    Main function to iterate through the dataset and process all videos.
    """
    print("Starting face extraction process...")
    
    print("Initializing face detector (this may take a moment)...")
    detector = MTCNN()
    print("Detector initialized.")
    
    # Define the video types to process
    data_types = {
        "original": "real",
        "Deepfakes": "fake",
        "Face2Face": "fake",
        "FaceSwap": "fake",
        "NeuralTextures": "fake"
    }

    for video_type, label in data_types.items():
        input_video_dir = VIDEO_DATA_ROOT / video_type
        output_face_dir = OUTPUT_FACE_DATA_ROOT / label

        if not input_video_dir.exists():
            print(f"Warning: Directory not found, skipping: {input_video_dir}")
            continue

        print(f"\nProcessing videos from: {video_type} (Label: {label})")
        
        video_files = list(input_video_dir.glob("*.mp4")) # Assuming videos are .mp4
        
        for video_file in tqdm(video_files, desc=f"Extracting faces from {video_type}"):
            extract_faces(video_file, output_face_dir, detector)
            
    print("\nFace extraction complete!")
    print(f"All extracted faces are saved in: {OUTPUT_FACE_DATA_ROOT}")


if __name__ == "__main__":
    main()
