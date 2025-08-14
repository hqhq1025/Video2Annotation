import json
import os
import math
from typing import List, Dict, Any

def load_scenes_with_conversation_summaries(file_path: str) -> List[Dict]:
    """
    Load the scenes data which includes timestamps, keyframe paths, summaries, and conversation summaries.
    
    Args:
        file_path (str): Path to the JSON file containing scene data.
        
    Returns:
        List[Dict]: A list of scene dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_all_frame_paths(frames_dir: str, total_frames: int) -> List[str]:
    """
    Get a list of all frame paths in order.
    
    Args:
        frames_dir (str): Directory containing the frame images.
        total_frames (int): Total number of frames to include.
        
    Returns:
        List[str]: A list of relative frame paths.
    """
    frame_paths = []
    for i in range(total_frames):
        frame_filename = f"frame_{i:05d}.jpg"
        # Store relative path as in the example format
        relative_path = os.path.join(os.path.basename(frames_dir), frame_filename) 
        frame_paths.append(relative_path)
    return frame_paths

def build_train_conversation(video_filename: str, 
                             all_frame_paths: List[str], 
                             scenes: List[Dict],
                             first_scene_summary: str = "Please appropriately describe what is happening in the video.") -> Dict[str, Any]:
    """
    Builds the train_conversation JSON object for a single video.
    
    Args:
        video_filename (str): The name of the video file.
        all_frame_paths (List[str]): List of all frame paths in order.
        scenes (List[Dict]): List of scene data with conversation_summary.
        first_scene_summary (str, optional): The initial prompt. Defaults to a generic one.
        
    Returns:
        Dict[str, Any]: A dictionary representing the train_conversation object.
    """
    images = []
    conversations = []
    
    # Add the fixed initial human instruction
    conversations.append({
        "from": "human",
        "value": first_scene_summary
    })
    
    # Create a mapping from end_frame_index to conversation_summary for easy lookup
    summary_map = {}
    for scene in scenes:
        end_ts = scene['end_timestamp']
        # Convert end timestamp to frame index (assuming 1fps)
        end_frame_index = int(round(end_ts)) # e.g., 9.166 -> 9, 29.625 -> 30 (but we check for rounding)
        # Use the English conversation_summary
        conv_summary = scene.get('conversation_summary', '')
        summary_map[end_frame_index] = conv_summary
        
    # Debug: print summary_map
    print(f"Summary map keys (end_frame_indices): {list(summary_map.keys())}")

    # Iterate through all frames
    total_frames = len(all_frame_paths)
    for i, frame_path in enumerate(all_frame_paths):
        images.append(frame_path)
        
        # Add human image input
        conversations.append({
            "from": "human",
            "value": "<image>"
        })
        
        # Determine if this frame is the last frame of any scene
        matched_summary = summary_map.get(i)
                
        if matched_summary is not None:
            # This is the last frame of a scene, output the conversation summary
            conversations.append({
                "from": "gpt",
                "value": f"<|response|> {matched_summary}"
            })
        else:
            # Regular frame, output silent
            conversations.append({
                "from": "gpt",
                "value": "<|silent|>"
            })
    
    # Add the final END_OF_STREAMING marker
    conversations.append({
        "from": "human",
        "value": "<|END_OF_STREAMING|>"
    })
    
    # Add the final summary (from the last scene)
    if scenes:
        last_scene = scenes[-1]
        last_summary = last_scene.get('conversation_summary', '[Final summary not available]')
        conversations.append({
            "from": "gpt",
            "value": f"<|response|> {last_summary}"
        })
    else:
        conversations.append({
            "from": "gpt",
            "value": "<|response|> [No scenes defined]"
        })

    return {
        "video": video_filename,
        "images": images,
        "conversations": conversations
    }

def generate_train_conversations(scenes_file: str,
                                frames_dir: str,
                                video_filename: str,
                                output_file: str,
                                total_frames: int):
    """
    Main function to generate the train_conversations.json file.
    
    Args:
        scenes_file (str): Path to the scenes JSON file with conversation summaries.
        frames_dir (str): Directory containing the extracted frames.
        video_filename (str): Name of the original video file.
        output_file (str): Path to the output JSON file.
        total_frames (int): Total number of frames extracted.
    """
    print("Loading scenes with conversation summaries...")
    scenes = load_scenes_with_conversation_summaries(scenes_file)
    print(f"Loaded {len(scenes)} scenes.")
    
    print("Getting list of all frame paths...")
    all_frame_paths = get_all_frame_paths(frames_dir, total_frames)
    print(f"Found {len(all_frame_paths)} frame paths.")
    
    print("Building train conversation data...")
    train_conversation = build_train_conversation(video_filename, all_frame_paths, scenes)
    
    print(f"Saving train conversation to {output_file}...")
    # Save as a list with one element, as per the example format
    final_output = [train_conversation]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print("Done.")

def main():
    """
    Main function to configure and run the generation process.
    """
    # --- Configuration ---
    scenes_file = "/Users/haoqing/Documents/Github/Video2Annotation/demo_output/scenes_with_conversation_summaries_en.json"
    frames_dir = "/Users/haoqing/Documents/Github/Video2Annotation/demo_output/demo_frames"
    video_filename = "Bad.Boys.II.2003__#01-11-16_01-14-00_label_A_E0.mp4"
    output_file = "/Users/haoqing/Documents/Github/Video2Annotation/demo_output/train_conversations.json"
    # Total frames from frame_00000.jpg to frame_00032.jpg is 33
    total_frames = 33
    
    # --- Execution ---
    generate_train_conversations(scenes_file, frames_dir, video_filename, output_file, total_frames)

if __name__ == "__main__":
    main()