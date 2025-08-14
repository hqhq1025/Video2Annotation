from http import HTTPStatus
import os
import base64
from openai import OpenAI

# Define the model to use as a constant
# Use environment variable DASHSCOPE_API_KEY, with a placeholder if not set
QWENVL_MODEL = "qwen-vl-plus" # Commonly available QwenVL model on DashScope

def encode_image_to_base64(image_path):
    """
    Encodes an image file to a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image, or None if an error occurred.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"Error encoding image {image_path} to base64: {e}")
        return None

def get_qwenvl_caption(image_path, prompt="Describe this image in detail."):
    """
    Gets a caption for an image using QwenVL model via DashScope API.

    Args:
        image_path (str): Path to the image file.
        prompt (str, optional): The prompt to send to the model. Defaults to "Describe this image in detail.".

    Returns:
        str: The generated caption, or None if an error occurred.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    # Encode the image to base64
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    # Initialize the OpenAI client for DashScope
    # Prioritize environment variable, fallback to a placeholder (won't work without real key)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY_PLACEHOLDER")
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    try:
        print(f"Calling {QWENVL_MODEL} for {image_path}...")
        completion = client.chat.completions.create(
            model=QWENVL_MODEL, # Use the defined QwenVL model
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}} # Pass the base64 encoded image
                    ]
                }
            ],
            temperature=0.8 # Revert to earlier temperature
        )
        # Check if the response is successful
        if completion.choices and completion.choices[0].message.content:
            caption = completion.choices[0].message.content
            print(f"Caption generated for {image_path}")
            return caption
        else:
            print(f"Warning: No caption returned for {image_path}")
            return None

    except Exception as e:
        print(f"Error calling QwenVL API ({QWENVL_MODEL}) for {image_path}: {e}")
        return None