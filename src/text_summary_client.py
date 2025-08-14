from http import HTTPStatus
import os
from openai import OpenAI

# Define the text model to use as a constant
# Use environment variable DASHSCOPE_API_KEY, with a placeholder if not set
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
    # Prioritize environment variable, fallback to a placeholder (won't work without real key)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY_PLACEHOLDER")
    client = OpenAI(
        api_key=api_key, # It's better to use environment variable
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Default prompt if none is provided, optimized for narrative scene description
    if not custom_prompt:
        custom_prompt = """
You are a talented storyteller and screenwriter. Your task is to create a **coherent, vivid, and engaging narrative** that describes a short video scene.

You will be given a sequence of detailed descriptions. Each description corresponds to a keyframe from the same continuous video clip. Your goal is to synthesize these descriptions into a **single, flowing paragraph** that tells the story of what happens in the entire scene, as if you were describing it to someone who hasn't seen the video.

**Instructions:**
1.  **Read all descriptions carefully.** They depict different moments from the same scene.
2.  **Identify the core setting, characters, actions, and mood.** What is the environment? Who or what is present? What are they doing? How does the scene feel (e.g., tense, joyful, mysterious)?
3.  **Craft a narrative.** Instead of listing details from each frame, weave them into a chronological and logical flow. Use transitional language (e.g., "then," "suddenly," "meanwhile," "as time passes") to connect moments if needed, but avoid being overly mechanical.
4.  **Prioritize coherence and storytelling.** The summary should read like a short story excerpt or a film critic's description of a scene. It should have a clear beginning, middle, and sense of progression or culmination.
5.  **Be descriptive and vivid.** Use evocative language to paint a picture, but keep it concise and focused.
6.  **Avoid Repetition.** Do not simply repeat information from every frame description. Synthesize and prioritize the most important elements for the overall narrative.

**Goal:** Create a summary so vivid and well-structured that it allows the reader to clearly visualize and understand the essence of the video scene.

**Keyframe Descriptions:**
"""

    # Build the full prompt by appending the captions
    full_prompt = custom_prompt + "\n"
    for i, caption in enumerate(captions_list):
        full_prompt += f"--- Frame {i+1} Description ---\n{caption}\n\n"

    try:
        print(f"Calling {QWEN_TEXT_MODEL} to summarize {len(captions_list)} caption(s)...")
        completion = client.chat.completions.create(
            model=QWEN_TEXT_MODEL, # Use the defined QwenVL model
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant and a skilled creative writer.'},
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