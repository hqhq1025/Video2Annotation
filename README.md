# Video to Annotation Tool

This tool processes videos to detect scene changes, extracts key frames for each scene, and generates natural language annotations for these scenes using the QwenVL model. The output is structured for training machine learning models.

## Features

- **Scene Change Detection**: Identifies significant visual transitions in a video.
- **Frame Extraction**: Extracts key frames corresponding to the start of each detected scene.
- **QwenVL Integration**: Generates descriptive, natural language captions for each scene using the QwenVL model.
- **Structured Output**: Produces a JSON file with video path, extracted frame paths, and generated annotations, following a predefined format suitable for training datasets.

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- FFmpeg (for video processing)
- QwenVL model API/library

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Video2Annotation.git
cd Video2Annotation

# (It's recommended to use a virtual environment)
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`

# Install dependencies (dependencies will be listed in requirements.txt)
pip install -r requirements.txt
```

## Usage

```bash
# Extract frames at 1 frame per second (default)
python src/video2annotation.py --extract-frames /path/to/your/video.mp4

# (Future) Process video for full annotation pipeline
# python src/video2annotation.py /path/to/your/video.mp4
```

This will process the video, extract frames, and save the output frames to a directory.

## Output Format

The tool generates a JSON file structured as follows:

```json
{
  "video": "/path/to/your/video.mp4",
  "images": [
    "/path/to/extracted/frame_00000.jpg",
    "/path/to/extracted/frame_00001.jpg",
    // ... paths to all key frames extracted for annotation
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<image>"
    },
    {
      "from": "gpt",
      "value": "Description of the first scene corresponding to the first frame."
    },
    {
      "from": "human",
      "value": "<image>"
    },
    {
      "from": "gpt",
      "value": "Description of the second scene corresponding to the second frame."
    }
    // ... alternating 'human' and 'gpt' entries for each scene/frame
  ]
}
```

## License

MIT License (or your chosen license)