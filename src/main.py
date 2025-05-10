# src/main.py
import os
import json
from dotenv import load_dotenv
import datetime

# It's good practice to handle imports from within the src package using relative imports
from .chromadb_manager import ChromaDBManager
from .llm_interface import StoryContextGenerator 

def initialize_system():
    """Initializes ChromaDB, loads data, and initializes the LLM interface."""
    print("--- System Initialization Started ---")
    load_dotenv() # Load environment variables from .env
    
    print("\nInitializing ChromaDB Manager...")
    chroma_manager = ChromaDBManager()
    
    print("\nLoading data into ChromaDB (will skip existing entries)...")
    data_file = os.getenv("SAMPLE_DATA_PATH", "data/sample_data.json")
    chroma_manager.load_data_from_json(json_file_path=data_file)
        
    print("\nInitializing LLM Story Context Generator (this may take a while for model loading)...")
    story_context_generator = StoryContextGenerator(chromadb_manager=chroma_manager)
    
    print("\n--- System Initialization Complete ---")
    return chroma_manager, story_context_generator

def save_output_to_json(output_data, prompt, output_dir="output"):
    """Saves the generated story elements to a JSON file"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a safe filename from the prompt (first 30 chars)
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30]).rstrip("_")
    
    # Create the full output data with the prompt included
    full_output = {
        "prompt": prompt,
        "timestamp": timestamp,
        "story_elements": output_data
    }
    
    # Create the filename
    filename = f"{output_dir}/story_{timestamp}_{safe_prompt}.json"
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    
    return filename

def main():
    chroma_manager, story_generator = initialize_system()

    while True:
        print("\n==================================================")
        # Changed prompt to English
        story_details_prompt = input("Enter story details (or type 'exit' to quit): ")
        print("==================================================")

        if not story_details_prompt.strip():
            # Changed message to English
            print("No story details entered. Please try again.")
            continue
        
        if story_details_prompt.lower() == 'exit':
            # Changed message to English
            print("Exiting application...")
            break

        # Changed message to English
        print(f"\nProcessing story idea: '{story_details_prompt}'...")
        print("AI (LLM) is deciding what information to retrieve from the database...")

        final_output_from_llm = story_generator.generate_story_elements(story_details_prompt)
        
        output_characters = []
        if final_output_from_llm.get("Characters"):
            for char_meta in final_output_from_llm["Characters"]:
                output_characters.append({
                    "character_id": char_meta.get("character_id", "N/A"),
                    "character_name": char_meta.get("character_name", "N/A"),
                    "character_profession": char_meta.get("character_profession", "N/A"),
                    "bio_detail": char_meta.get("bio_detail", "N/A")
                })

        output_locations = []
        if final_output_from_llm.get("Locations"):
            for loc_meta in final_output_from_llm["Locations"]:
                output_locations.append({
                    "location_id": loc_meta.get("location_id", "N/A"),
                    "location_name": loc_meta.get("location_name", "N/A"),
                    "location_type": loc_meta.get("location_type", "N/A"),
                    "location_detail": loc_meta.get("location_detail", "N/A")
                })
            
        structured_output_for_display = {
            "Characters": output_characters,
            "Locations": output_locations
        }
        
        # Save the output to a JSON file
        output_file = save_output_to_json(structured_output_for_display, story_details_prompt)
        
        # Changed message to English
        print("\n--- AI Output (JSON format) ---")
        print(json.dumps(structured_output_for_display, indent=2, ensure_ascii=False)) # ensure_ascii=False is good for Thai names if they were mixed
        print("-----------------------------")
        print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Changed message to English
        print(f"A critical error occurred in the main application: {e}")
        import traceback
        traceback.print_exc()