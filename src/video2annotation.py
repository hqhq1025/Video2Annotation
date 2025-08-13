import cv2
import os
import argparse
import json
import numpy as np

def extract_frames(video_path, output_dir, fps=1):
    """
    Extracts frames from a video at a specified frame rate.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        fps (int, optional): Frames per second to extract. Defaults to 1.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {video_fps}")
    print(f"Total Frames: {total_frames}")

    # Calculate the interval in frames between extractions
    frame_interval = int(video_fps / fps) if fps > 0 else 1
    if frame_interval == 0:
        frame_interval = 1 # Avoid division by zero, extract every frame if fps is very high

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this frame should be saved
        if frame_count % frame_interval == 0:
            # Construct output filename
            output_filename = os.path.join(output_dir, f"frame_{saved_frame_count:05d}.jpg")
            
            # Save the frame
            success = cv2.imwrite(output_filename, frame)
            if success:
                print(f"Saved frame {saved_frame_count} to {output_filename}")
                saved_frame_count += 1
            else:
                print(f"Error: Could not save frame {saved_frame_count}")

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Finished extracting {saved_frame_count} frames.")


def detect_scene_changes(video_path, threshold=0.3):
    """
    Detects scene changes in a video using frame difference method.

    Args:
        video_path (str): Path to the input video file.
        threshold (float, optional): Threshold for detecting a scene change based on frame difference. Defaults to 0.3.

    Returns:
        list: A list of dictionaries containing 'timestamp' and 'frame_number' for each detected scene change.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    scene_changes = []
    prev_frame = None
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to grayscale for easier comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Calculate absolute difference between current and previous frame
            frame_diff = cv2.absdiff(prev_frame, gray_frame)
            
            # Calculate mean difference
            mean_diff = np.mean(frame_diff)
            
            # Normalize the difference (assuming pixel values are 0-255)
            normalized_diff = mean_diff / 255.0
            
            # If difference is above threshold, it's a scene change
            if normalized_diff > threshold:
                timestamp = frame_number / cap.get(cv2.CAP_PROP_FPS)
                scene_changes.append({
                    "timestamp": timestamp,
                    "frame_number": frame_number
                })
                print(f"Scene change detected at frame {frame_number}, timestamp {timestamp:.2f}s (diff: {normalized_diff:.4f})")
        
        prev_frame = gray_frame
        frame_number += 1
    
    cap.release()
    print(f"Detected {len(scene_changes)} scene changes.")
    return scene_changes


def save_scene_changes_to_json(scene_changes, output_file):
    """
    Saves the list of scene changes to a JSON file.

    Args:
        scene_changes (list): List of scene change dictionaries.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(scene_changes, f, indent=4)
    print(f"Scene changes saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Video to Annotation Tool")
    parser.add_argument("--extract-frames", type=str, help="Path to the input video file for frame extraction")
    parser.add_argument("--output-dir", type=str, help="Directory to save extracted frames (default: ./extracted_frames)")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract (default: 1)")
    
    parser.add_argument("--detect-scenes", type=str, help="Path to the input video file for scene change detection")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for scene change detection (default: 0.3)")
    parser.add_argument("--scene-output", type=str, help="Output JSON file for scene changes (default: ./scene_changes.json)")

    args = parser.parse_args()

    if args.extract_frames:
        video_path = args.extract_frames
        output_dir = args.output_dir if args.output_dir else "./extracted_frames"
        extract_frames(video_path, output_dir, args.fps)
    elif args.detect_scenes:
        video_path = args.detect_scenes
        scene_changes = detect_scene_changes(video_path, args.threshold)
        output_file = args.scene_output if args.scene_output else "./scene_changes.json"
        save_scene_changes_to_json(scene_changes, output_file)
    else:
        print("This is a placeholder for the main script. Use --extract-frames or --detect-scenes.")

if __name__ == "__main__":
    main()