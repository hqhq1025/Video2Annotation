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


def find_keyframes_for_scenes(scenes_file, frames_dir):
    """
    Finds all keyframe paths for each scene based on its start and end timestamps.
    Assumes frames are named frame_00000.jpg, frame_00001.jpg, etc., 
    corresponding to 1 frame per second.

    Args:
        scenes_file (str): Path to the JSON file containing scene data.
        frames_dir (str): Path to the directory containing extracted frames.

    Returns:
        list: A list of dictionaries, each containing scene info and 'keyframe_paths'.
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
        end_ts = scene['end_timestamp']
        
        # Find the frame index range corresponding to the scene timestamps
        # Assuming 1 frame per second, the index is the integer part of the timestamp
        start_frame_index = int(start_ts)
        end_frame_index = int(end_ts)
        
        # Collect all frame paths within this range (inclusive)
        keyframe_paths = []
        for i in range(start_frame_index, end_frame_index + 1):
            frame_filename = f"frame_{i:05d}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            # Check if the frame file actually exists before adding
            if os.path.exists(frame_path):
                keyframe_paths.append(frame_path)
            # else:
            #     print(f"Warning: Expected frame {frame_path} not found.")
            
        # Add the keyframe paths to the scene data
        annotated_scene = scene.copy()
        annotated_scene['keyframe_paths'] = keyframe_paths
        annotated_scenes.append(annotated_scene)
        print(f"Scene ({start_ts:.2f}s - {end_ts:.2f}s): Found {len(keyframe_paths)} keyframes.")
        
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

def save_frame_captions_to_json(frame_captions, output_file):
    """
    Saves the dictionary of frame paths to captions to a JSON file.

    Args:
        frame_captions (dict): A dictionary mapping frame paths to captions.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(frame_captions, f, indent=4, ensure_ascii=False)
    print(f"Frame captions saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Video to Annotation Tool")
    parser.add_argument("--extract-frames", type=str, help="Path to the input video file for frame extraction")
    parser.add_argument("--output-dir", type=str, help="Directory to save extracted frames (default: ./extracted_frames)")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract (default: 1)")
    
    parser.add_argument("--detect-scenes", type=str, help="Path to the input video file for scene detection")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for scene change detection (default: 0.3)")
    parser.add_argument("--scene-output", type=str, help="Output JSON file for scenes (default: ./scenes.json)")
    parser.add_argument("--min-scene-duration", type=float, default=2.0, help="Minimum duration (seconds) for a scene (default: 2.0)")

    # Argument for finding keyframes (ALL keyframes for each scene)
    parser.add_argument("--find-all-keyframes", type=str, help="Path to the JSON file containing scene data for keyframe lookup")
    parser.add_argument("--frames-dir", type=str, help="Path to the directory containing extracted frames")
    parser.add_argument("--all-keyframes-output", type=str, help="Output JSON file with ALL keyframe paths for each scene (default: ./scenes_with_all_keyframes.json)")

    # Argument for generating QwenVL captions
    parser.add_argument("--generate-captions", type=str, help="Path to the JSON file containing scenes with ALL keyframe paths")
    parser.add_argument("--frame-captions-output", type=str, help="Output JSON file with frame path to caption mapping (default: ./frame_captions.json)")
    parser.add_argument("--caption-prompt", type=str, default="Describe this image in detail.", help="Prompt to use for QwenVL captioning (default: 'Describe this image in detail.')")

    # Argument for summarizing scenes using a text model
    parser.add_argument("--summarize-scenes", type=str, help="Path to the JSON file containing scenes with ALL keyframe paths")
    parser.add_argument("--frame-captions", type=str, help="Path to the JSON file containing frame path to caption mapping")
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
    elif args.find_all_keyframes:
        if not args.frames_dir:
            print("Error: --frames-dir is required for --find-all-keyframes.")
            return
        scenes_with_all_keyframes = find_keyframes_for_scenes(args.find_all_keyframes, args.frames_dir)
        output_file = args.all_keyframes_output if args.all_keyframes_output else "./scenes_with_all_keyframes.json"
        save_scenes_to_json(scenes_with_all_keyframes, output_file)
        print(f"All keyframe paths added to scenes. Output saved to {output_file}")
    elif args.generate_captions:
        if not args.frame_captions_output:
             args.frame_captions_output = "./frame_captions.json"
             
        try:
            with open(args.generate_captions, 'r') as f:
                scenes = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find scenes file {args.generate_captions}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {args.generate_captions}")
            return

        print(f"Generating captions for all keyframes in {len(scenes)} scenes using QwenVL...")
        frame_captions_dict = {}
        
        # 1. Collect ALL unique frame paths from all scenes
        all_unique_frame_paths = set()
        for scene in scenes:
            keyframe_paths = scene.get('keyframe_paths', [])
            all_unique_frame_paths.update(keyframe_paths)
        
        total_frames = len(all_unique_frame_paths)
        print(f"Found {total_frames} unique frames to process.")
        
        # 2. Generate captions for each unique frame
        for i, frame_path in enumerate(sorted(list(all_unique_frame_paths))):
            if not os.path.exists(frame_path):
                print(f"Warning: Frame not found at {frame_path}, skipping.")
                continue

            print(f"Processing frame {i+1}/{total_frames}: {frame_path}")
            caption = get_qwenvl_caption(frame_path, args.caption_prompt)
            
            # Populate the frame_captions_dict
            if caption is not None:
                frame_captions_dict[frame_path] = caption
            # else: log or handle error
        
        # 3. Save the frame-level captions
        save_frame_captions_to_json(frame_captions_dict, args.frame_captions_output)
        print(f"Frame-level captions saved to {args.frame_captions_output}")

    elif args.summarize_scenes:
        if not args.frame_captions:
            print("Error: --frame-captions is required for --summarize-scenes.")
            return
        try:
            with open(args.summarize_scenes, 'r') as f:
                scenes = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find scenes file {args.summarize_scenes}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {args.summarize_scenes}")
            return
            
        try:
            with open(args.frame_captions, 'r') as f:
                frame_captions_dict = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find frame captions file {args.frame_captions}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {args.frame_captions}")
            return

        print(f"Summarizing captions for {len(scenes)} scenes using Qwen Text Model...")
        for i, scene in enumerate(scenes):
            # Get the list of keyframe paths for this scene
            keyframe_paths = scene.get('keyframe_paths', [])
            if not keyframe_paths:
                print(f"Skipping scene {i+1}: No keyframe paths found.")
                scene['summary'] = None
                continue

            # Get the captions for all keyframes from the frame_captions_dict
            captions_list = []
            for kf_path in keyframe_paths:
                cap = frame_captions_dict.get(kf_path)
                if cap:
                    captions_list.append(cap)
                # else: 
                #     print(f"Warning: No caption found for keyframe {kf_path} in scene {i+1}")
            
            if not captions_list:
                print(f"Skipping scene {i+1}: No captions found for any keyframes.")
                scene['summary'] = None
                continue
            
            print(f"Processing scene {i+1}/{len(scenes)} with {len(captions_list)} captions...")
            summary = get_qwen_text_summary(captions_list, args.summary_prompt)
            scene['summary'] = summary
        
        output_file = args.summary_output if args.summary_output else "./scenes_with_summaries.json"
        save_scenes_to_json(scenes, output_file)
        print(f"Scene summaries generated and saved to {output_file}")
    else:
        print("This is a placeholder for the main script. Use --extract-frames, --detect-scenes, --find-all-keyframes, --generate-captions, or --summarize-scenes.")

if __name__ == "__main__":
    main()