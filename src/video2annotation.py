import cv2
import os
import argparse
import json
import numpy as np

# Import the QwenVL client function
from qwenvl_client import get_qwenvl_caption
# Import the text summary client function
from text_summary_client import get_qwen_text_summary

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


def merge_short_scenes(scenes, min_duration=2.0):
    """
    Merges scenes that are shorter than min_duration into adjacent scenes.

    Args:
        scenes (list): List of scene dictionaries.
        min_duration (float, optional): Minimum duration for a scene in seconds. Defaults to 2.0.

    Returns:
        list: The list of scenes after merging short ones.
    """
    if len(scenes) <= 1:
        return scenes

    # Work on a copy to avoid modifying the original list during iteration
    merged_scenes = scenes.copy()
    i = 0
    while i < len(merged_scenes):
        scene = merged_scenes[i]
        duration = scene['end_timestamp'] - scene['start_timestamp']
        
        # Check if current scene is too short
        if duration < min_duration:
            print(f"Scene starting at {scene['start_timestamp']:.2f}s is short (Duration: {duration:.2f}s). Merging...")
            
            # Case 1: First scene
            if i == 0:
                # Merge into the next scene
                if i + 1 < len(merged_scenes):
                    merged_scenes[i + 1]['start_timestamp'] = scene['start_timestamp']
                    merged_scenes[i + 1]['start_frame'] = scene['start_frame']
                    print(f"  Merged into next scene (now starts at {merged_scenes[i + 1]['start_timestamp']:.2f}s).")
                    merged_scenes.pop(i) # Remove current scene
                    # Do not increment i, as the next scene has shifted to index i
                    continue 
                else:
                    # Only one scene, nothing to merge with. This case should not happen due to len>1 check.
                    pass
            
            # Case 2: Last scene
            elif i == len(merged_scenes) - 1:
                # Merge into the previous scene
                if i - 1 >= 0:
                    merged_scenes[i - 1]['end_timestamp'] = scene['end_timestamp']
                    print(f"  Merged into previous scene (now ends at {merged_scenes[i - 1]['end_timestamp']:.2f}s).")
                    merged_scenes.pop(i) # Remove current scene
                    # Do not increment i, as we need to check the new scene at this index
                    continue
            
            # Case 3: Middle scene
            else:
                # Merge into the next scene (or previous, but next is simpler)
                if i + 1 < len(merged_scenes):
                    merged_scenes[i + 1]['start_timestamp'] = scene['start_timestamp']
                    merged_scenes[i + 1]['start_frame'] = scene['start_frame']
                    print(f"  Merged into next scene (now starts at {merged_scenes[i + 1]['start_timestamp']:.2f}s).")
                    merged_scenes.pop(i) # Remove current scene
                    # Do not increment i
                    continue
                # This else clause is a fallback, though unlikely to be reached if i+1 < len
                # elif i - 1 >= 0:
                #     merged_scenes[i - 1]['end_timestamp'] = scene['end_timestamp']
                #     merged_scenes.pop(i)
                #     continue
            
        # If scene was not merged, or was the last scene and not short, move to next
        i += 1
        
    print(f"Merged short scenes. Original: {len(scenes)}, Final: {len(merged_scenes)}")
    return merged_scenes


def find_keyframe_for_scenes(scenes_file, frames_dir):
    """
    Finds the keyframe path for each scene based on its start timestamp.
    Assumes frames are named frame_00000.jpg, frame_00001.jpg, etc., 
    corresponding to 1 frame per second.

    Args:
        scenes_file (str): Path to the JSON file containing scene data.
        frames_dir (str): Path to the directory containing extracted frames.

    Returns:
        list: A list of dictionaries, each containing scene info and 'keyframe_path'.
    """
    try:
        with open(scenes_file, 'r') as f:
            scenes = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find scenes file {scenes_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {scenes_file}")
        return []

    annotated_scenes = []
    for scene in scenes:
        start_ts = scene['start_timestamp']
        # Find the frame index corresponding to the start timestamp
        # Assuming 1 frame per second, the index is the integer part of the timestamp
        frame_index = int(start_ts) # This effectively floors the timestamp
        
        # Construct the expected frame filename
        frame_filename = f"frame_{frame_index:05d}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)
        
        # Check if the frame file actually exists
        if not os.path.exists(frame_path):
            print(f"Warning: Keyframe {frame_path} for scene starting at {start_ts:.2f}s not found.")
            frame_path = None # Or handle the error as appropriate
            
        # Add the keyframe path to the scene data
        annotated_scene = scene.copy()
        annotated_scene['keyframe_path'] = frame_path
        annotated_scenes.append(annotated_scene)
        
    return annotated_scenes


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
    parser.add_argument("--min-scene-duration", type=float, default=2.0, help="Minimum duration (seconds) for a scene (default: 2.0)")

    # Argument for finding keyframes
    parser.add_argument("--find-keyframes", type=str, help="Path to the JSON file containing scene data for keyframe lookup")
    parser.add_argument("--frames-dir", type=str, help="Path to the directory containing extracted frames")
    parser.add_argument("--keyframe-output", type=str, help="Output JSON file with keyframe paths (default: ./scenes_with_keyframes.json)")

    # Argument for generating QwenVL captions
    parser.add_argument("--generate-captions", type=str, help="Path to the JSON file containing scenes with keyframe paths")
    parser.add_argument("--captions-output", type=str, help="Output JSON file with QwenVL captions (default: ./scenes_with_captions.json)")
    parser.add_argument("--caption-prompt", type=str, default="Describe this image in detail.", help="Prompt to use for QwenVL captioning (default: 'Describe this image in detail.')")

    # Argument for summarizing scenes using a text model
    parser.add_argument("--summarize-scenes", type=str, help="Path to the JSON file containing scenes with captions to be summarized")
    parser.add_argument("--summary-output", type=str, help="Output JSON file with summarized scenes (default: ./scenes_with_summaries.json)")
    parser.add_argument("--summary-prompt", type=str, help="Custom prompt for the text summarization model")

    args = parser.parse_args()

    if args.extract_frames:
        video_path = args.extract_frames
        output_dir = args.output_dir if args.output_dir else "./extracted_frames"
        extract_frames(video_path, output_dir, args.fps)
    elif args.detect_scenes:
        video_path = args.detect_scenes
        scenes = detect_scenes_and_timestamps(video_path, args.threshold)
        # Apply post-processing to merge short scenes
        scenes = merge_short_scenes(scenes, args.min_scene_duration)
        output_file = args.scene_output if args.scene_output else "./scenes.json"
        save_scenes_to_json(scenes, output_file)
    elif args.find_keyframes:
        if not args.frames_dir:
            print("Error: --frames-dir is required for --find-keyframes.")
            return
        scenes_with_keyframes = find_keyframe_for_scenes(args.find_keyframes, args.frames_dir)
        output_file = args.keyframe_output if args.keyframe_output else "./scenes_with_keyframes.json"
        save_scenes_to_json(scenes_with_keyframes, output_file)
        print(f"Keyframe paths added to scenes. Output saved to {output_file}")
    elif args.generate_captions:
        try:
            with open(args.generate_captions, 'r') as f:
                scenes = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find scenes file {args.generate_captions}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {args.generate_captions}")
            return

        print(f"Generating captions for {len(scenes)} scenes using QwenVL...")
        for i, scene in enumerate(scenes):
            keyframe_path = scene.get('keyframe_path')
            if not keyframe_path or not os.path.exists(keyframe_path):
                print(f"Skipping scene {i+1}: Keyframe not found at {keyframe_path}")
                scene['caption'] = None
                continue

            print(f"Processing scene {i+1}/{len(scenes)}: {keyframe_path}")
            caption = get_qwenvl_caption(keyframe_path, args.caption_prompt)
            scene['caption'] = caption
        
        output_file = args.captions_output if args.captions_output else "./scenes_with_captions.json"
        save_scenes_to_json(scenes, output_file)
        print(f"Captions generated and saved to {output_file}")
    elif args.summarize_scenes:
        try:
            with open(args.summarize_scenes, 'r') as f:
                scenes = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find scenes file {args.summarize_scenes}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {args.summarize_scenes}")
            return

        print(f"Summarizing captions for {len(scenes)} scenes using Qwen Text Model...")
        for i, scene in enumerate(scenes):
            # Get the list of captions for this scene.
            # For now, we assume it's a single 'caption' field.
            # In the future, this could be a list of 'captions'.
            original_caption = scene.get('caption')
            if not original_caption:
                print(f"Skipping scene {i+1}: No caption found.")
                scene['summary'] = None
                continue

            # Wrap the single caption in a list to match the expected input format
            captions_list = [original_caption]
            
            print(f"Processing scene {i+1}/{len(scenes)}...")
            summary = get_qwen_text_summary(captions_list, args.summary_prompt)
            scene['summary'] = summary
        
        output_file = args.summary_output if args.summary_output else "./scenes_with_summaries.json"
        save_scenes_to_json(scenes, output_file)
        print(f"Scene summaries generated and saved to {output_file}")
    else:
        print("This is a placeholder for the main script. Use --extract-frames, --detect-scenes, --find-keyframes, --generate-captions, or --summarize-scenes.")

if __name__ == "__main__":
    main()