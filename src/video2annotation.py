import cv2
import os
import argparse

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


def main():
    parser = argparse.ArgumentParser(description="Video to Annotation Tool")
    parser.add_argument("--extract-frames", type=str, help="Path to the input video file for frame extraction")
    parser.add_argument("--output-dir", type=str, help="Directory to save extracted frames (default: ./extracted_frames)")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract (default: 1)")

    args = parser.parse_args()

    if args.extract_frames:
        video_path = args.extract_frames
        output_dir = args.output_dir if args.output_dir else "./extracted_frames"
        extract_frames(video_path, output_dir, args.fps)
    else:
        print("This is a placeholder for the main script. Use --extract-frames to extract frames from a video.")

if __name__ == "__main__":
    main()