from http import HTTPStatus
import os
from openai import OpenAI

# Define the text model to use as a constant
QWEN_TEXT_MODEL = "qwen-plus" # Qwen-Plus text model on DashScope

def get_qwen_text_summary(captions_list, custom_prompt=None):
    """
    Generates a summary from a list of captions using Qwen text model via DashScope API.

    Args:
        captions_list (list[str]): A list of caption strings to be summarized.
        custom_prompt (str, optional): A custom prompt to override the default one.

    Returns:
        str: The generated summary, or None if an error occurred.
    """
    if not captions_list:
        print("Error: No captions provided for summarization.")
        return None

    # Initialize the OpenAI client for DashScope
    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY", "sk-07809ce5885f4fc3aefd07b0ca0e1e11"), # It's better to use environment variable
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Default prompt if none is provided
    if not custom_prompt:
        custom_prompt = """
You are an AI assistant tasked with creating a comprehensive summary of a video scene.
You will be given one or more detailed descriptions of keyframes from that scene.
Your goal is to synthesize these descriptions into a single, coherent, and detailed paragraph that accurately represents the entire scene.

Instructions:
1.  Carefully read all the provided keyframe descriptions.
2.  Identify the core setting, main subjects, and key actions or events.
3.  If there are multiple descriptions, look for how the scene evolves or what different perspectives they offer.
4.  Combine the information to form a unified narrative of the scene.
5.  Be detailed and descriptive, but avoid simply concatenating the input texts.
6.  The summary should read as if it describes the scene as a whole, not as a collection of separate images.

Keyframe Description(s):
"""

    # Build the full prompt by appending the captions
    full_prompt = custom_prompt + "\n"
    for i, caption in enumerate(captions_list):
        full_prompt += f"--- Description {i+1} ---\n{caption}\n\n"

    try:
        print(f"Calling {QWEN_TEXT_MODEL} to summarize {len(captions_list)} caption(s)...")
        completion = client.chat.completions.create(
            model=QWEN_TEXT_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': full_prompt}
            ],
            temperature=0.7 # Moderate temperature for balanced creativity and relevance
        )
        # Check if the response is successful
        if completion.choices and completion.choices[0].message.content:
            summary = completion.choices[0].message.content.strip()
            print(f"Summary generated.")
            return summary
        else:
            print(f"Warning: No summary returned by the model.")
            return None

    except Exception as e:
        print(f"Error calling Qwen Text API ({QWEN_TEXT_MODEL}): {e}")
        return None