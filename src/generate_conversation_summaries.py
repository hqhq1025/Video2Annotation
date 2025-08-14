import json
import os
from typing import List, Dict, Any

# Import the text summary client function for LLM interaction
# We'll use the same Qwen model for consistency
from text_summary_client import get_qwen_text_summary # Assuming this function can be reused or adapted

def load_scenes_with_summaries(file_path: str) -> List[Dict]:
    """
    Load the scenes data which includes timestamps, keyframe paths, and summaries.
    
    Args:
        file_path (str): Path to the JSON file containing scene data.
        
    Returns:
        List[Dict]: A list of scene dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_transition_prompt(prev_summary: str, current_summary: str) -> List[str]:
    """
    Generates an improved prompt for the LLM to create a detailed, English conversational transition summary.
    Inspired by generate_concat_annotations.py.
    
    Args:
        prev_summary (str): Summary of the previous scene.
        current_summary (str): Summary of the current scene.
        
    Returns:
        List[str]: A list containing the prompt string, suitable for get_qwen_text_summary.
    """
    # Ensure summaries are strings
    prev_summary = str(prev_summary) if prev_summary is not None else ""
    current_summary = str(current_summary) if current_summary is not None else ""

    prompt = f"""You are a video concatenation description assistant. You will receive:
1. **History**: A description of the previous video scene.  
2. **Current**: A description of the current video scene.

Your task is to combine **Current** with **History** to produce one **coherent**, **natural**, and **detailed English** paragraph summary (2-4 sentences) in a continuous storytelling style. The summary will be used for training a video understanding model to generate descriptions at scene boundaries. It should:

- Seamlessly connect past and present content as a single narrative, referencing the previous scene to create smooth transitions between scenes.
- Focus on describing **what has changed** or **what's new** in the current scene, while maintaining context from the previous scene.
- When there are contextual connections between scenes, clearly describe what has been added, changed, or is being done differently based on the previous content.
- **Do not** repeat objects, settings, or details already mentioned in **History** unless necessary for context.
- **Avoid** any atmospheric, emotional, or subjective commentary; describe the visual content **objectively**.
- **Do not** start with "This video…" or similar phrases.
- If **History** is empty, simply summarize **Current** on its own.
- Use varied and natural transition expressions instead of always using "following". Examples include:
  * "After [previous action], the scene shifts to..."
  * "Continuing from the previous scene where [brief recap], we now see..."
  * "Building upon the earlier scenes of [brief recap], the current segment presents..."
  * "Transitioning from [previous setting], the focus now moves to..."
  * "With the completion of [previous activity], attention turns to..."
  * "The scene then changes to show..."
  * "Subsequently, the setting shifts to..."
- **Avoid** repetitive transitions or descriptions that simply restate what was already described in previous segments.
- **Focus** on how the current scene builds upon or differs from previous scenes rather than restating them.
- **Create** natural narrative flow by emphasizing the progression of activities or changes in setting.

Examples of good transitions:
- "After arranging objects in a box, the scene shifts to someone preparing a beverage in a kitchen."
- "Continuing from the previous scene where items were organized into bags, the person now moves to a different area to tidy up a pair of shoes."
- "Building upon the previous scenes of organizing items in the kitchen and packing belongings, the current segment shows a person carefully tying their shoelaces."
- "With the completion of organizing personal items into handbags, the focus now moves to a more personal grooming activity as the scene shows someone attending to their footwear."

---
### Input

History:
{prev_summary}

Current:
{current_summary}

### Output
A single paragraph summary (2-4 sentences) in natural storytelling style, written in **English**, highlighting changes and maintaining narrative flow while avoiding repetition. No extra text."""

    return [prompt]

def generate_conversation_summaries(scenes: List[Dict], custom_prompt_func=None) -> List[Dict]:
    """
    Generates conversational summaries for each scene.
    
    For the first scene, the conversational summary is the same as its original summary.
    For subsequent scenes, an LLM is used to generate a transition summary.
    
    Args:
        scenes (List[Dict]): List of scene data dictionaries.
        custom_prompt_func (function, optional): A function to generate custom prompts.
        
    Returns:
        List[Dict]: The list of scenes, each now with an added 'conversation_summary' field.
    """
    if not scenes:
        return scenes

    if custom_prompt_func is None:
        custom_prompt_func = generate_transition_prompt

    # Add 'conversation_summary' to each scene
    for i, scene in enumerate(scenes):
        if i == 0:
            # For the first scene, we still need an English summary.
            # For now, we will indicate it needs translation/processing.
            # A better approach might be to translate the first summary to English in a separate step,
            # or use a dedicated function. For this iteration, we'll flag it.
            first_summary = scene.get('summary', '')
            if first_summary:
                # Simple placeholder: indicate it's the first scene and should be in English.
                # A real implementation might call a translation model here.
                scene['conversation_summary'] = f"[START_OF_VIDEO] {first_summary}" # TODO: Translate to English
            else:
                scene['conversation_summary'] = "[START_OF_VIDEO] [No initial summary available]"
            print(f"Scene {i}: Assigned initial summary placeholder for conversation_summary.")
        else:
            prev_scene = scenes[i-1]
            prev_summary = prev_scene.get('summary', '')
            current_summary = scene.get('summary', '')
            
            if not prev_summary or not current_summary:
                print(f"Warning: Missing summary for scene {i-1} or {i}. Using '[Data Missing]' as conversation_summary.")
                scene['conversation_summary'] = "[Data Missing]"
                continue

            # Generate the prompt for the LLM
            try:
                prompt_list = custom_prompt_func(prev_summary, current_summary)
                print(f"Scene {i}: Calling LLM to generate English conversational summary...")
                # Use the existing get_qwen_text_summary function
                conversation_summary = get_qwen_text_summary(prompt_list)
                
                if conversation_summary:
                    scene['conversation_summary'] = conversation_summary
                    print(f"Scene {i}: English conversational summary generated.")
                else:
                    print(f"Warning: LLM returned no summary for scene {i}. Using '[LLM Failed]' as conversation_summary.")
                    scene['conversation_summary'] = "[LLM Failed]"
                    
            except Exception as e:
                print(f"Error generating conversational summary for scene {i}: {e}")
                scene['conversation_summary'] = "[Generation Error]"
                
    return scenes

def save_scenes_with_conversation_summaries(scenes: List[Dict], output_file: str):
    """
    Saves the list of scenes with added conversation summaries to a JSON file.
    
    Args:
        scenes (List[Dict]): List of scene data dictionaries.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    print(f"Scenes with conversation summaries saved to {output_file}")

def main():
    """
    Main function to orchestrate the generation of conversation summaries.
    """
    # --- Configuration ---
    # Input file (the one with scene summaries generated by our pipeline)
    # TODO: Replace with the correct path to your latest scenes summary file
    input_file = "/Users/haoqing/Documents/Github/Video2Annotation/demo_output/scenes_with_summaries_rolled_back_prompt.json"
    
    # Output file for scenes with added conversation summaries
    output_file = "/Users/haoqing/Documents/Github/Video2Annotation/demo_output/scenes_with_conversation_summaries_en.json"
    
    # --- Processing ---
    print("Loading scenes with summaries...")
    scenes = load_scenes_with_summaries(input_file)
    print(f"Loaded {len(scenes)} scenes.")
    
    if not scenes:
        print("No scenes found in the input file. Exiting.")
        return

    print("Generating English conversational summaries for each scene...")
    scenes_with_conv_summaries = generate_conversation_summaries(scenes)
    
    print("Saving results...")
    save_scenes_with_conversation_summaries(scenes_with_conv_summaries, output_file)
    
    print("Done.")

if __name__ == "__main__":
    main()