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


def detect_scenes_and_timestamps(video_path, threshold=0.3):
    """
    Detects scenes in a video and records start and end timestamps for each scene.

    Args:
        video_path (str): Path to the input video file.
        threshold (float, optional): Threshold for detecting a scene change. Defaults to 0.3.

    Returns:
        list: A list of dictionaries, each containing 'start_timestamp', 'end_timestamp', and 'start_frame'.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    scenes = []
    prev_frame = None
    frame_number = 0
    current_scene_start_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video, record the last scene
            if current_scene_start_frame <= frame_number:
                start_timestamp = current_scene_start_frame / cap.get(cv2.CAP_PROP_FPS)
                end_timestamp = (frame_number - 1) / cap.get(cv2.CAP_PROP_FPS)
                scenes.append({
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "start_frame": current_scene_start_frame
                })
            break
            
        # Convert frame to grayscale for easier comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        is_scene_change = False
        if prev_frame is not None:
            # Calculate absolute difference between current and previous frame
            frame_diff = cv2.absdiff(prev_frame, gray_frame)
            
            # Calculate mean difference
            mean_diff = np.mean(frame_diff)
            
            # Normalize the difference (assuming pixel values are 0-255)
            normalized_diff = mean_diff / 255.0
            
            # If difference is above threshold, it's a scene change
            if normalized_diff > threshold:
                is_scene_change = True
        
        if is_scene_change:
            # Record the previous scene
            start_timestamp = current_scene_start_frame / cap.get(cv2.CAP_PROP_FPS)
            end_timestamp = (frame_number - 1) / cap.get(cv2.CAP_PROP_FPS)
            scenes.append({
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "start_frame": current_scene_start_frame
            })
            print(f"Scene: {len(scenes)} - Start: {start_timestamp:.2f}s, End: {end_timestamp:.2f}s (Frames {current_scene_start_frame}-{frame_number-1})")
            
            # Start a new scene
            current_scene_start_frame = frame_number
            
        prev_frame = gray_frame
        frame_number += 1
    
    cap.release()
    print(f"Detected {len(scenes)} scenes.")
    return scenes


def save_scenes_to_json(scenes, output_file):
    """
    Saves the list of scenes with timestamps to a JSON file.

    Args:
        scenes (list): List of scene dictionaries.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(scenes, f, indent=4)
    print(f"Scenes saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Video to Annotation Tool")
    parser.add_argument("--extract-frames", type=str, help="Path to the input video file for frame extraction")
    parser.add_argument("--output-dir", type=str, help="Directory to save extracted frames (default: ./extracted_frames)")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract (default: 1)")
    
    parser.add_argument("--detect-scenes", type=str, help="Path to the input video file for scene detection")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for scene change detection (default: 0.3)")
    parser.add_argument("--scene-output", type=str, help="Output JSON file for scenes (default: ./scenes.json)")

    args = parser.parse_args()

    if args.extract_frames:
        video_path = args.extract_frames
        output_dir = args.output_dir if args.output_dir else "./extracted_frames"
        extract_frames(video_path, output_dir, args.fps)
    elif args.detect_scenes:
        video_path = args.detect_scenes
        scenes = detect_scenes_and_timestamps(video_path, args.threshold)
        output_file = args.scene_output if args.scene_output else "./scenes.json"
        save_scenes_to_json(scenes, output_file)
    else:
        print("This is a placeholder for the main script. Use --extract-frames or --detect-scenes.")

if __name__ == "__main__":
    main()