# src/main.py
import os
import json
from dotenv import load_dotenv

# It's good practice to handle imports from within the src package using relative imports
from .chromadb_manager import ChromaDBManager
from .llm_interface import StoryContextGenerator 

def initialize_system():
    """Initializes ChromaDB, loads data, and initializes the LLM interface."""
    print("--- System Initialization Started ---")
    load_dotenv() # Load environment variables from .env
    
    print("\nInitializing ChromaDB Manager...")
    chroma_manager = ChromaDBManager()
    
    # Simplified data loading: always attempt to load from JSON.
    # The add_X methods in ChromaDBManager now prevent duplicates.
    print("\nLoading data into ChromaDB (will skip existing entries)...")
    data_file = os.getenv("SAMPLE_DATA_PATH", "data/sample_data.json") # Assuming .env might define this
    chroma_manager.load_data_from_json(json_file_path=data_file)
        
    print("\nInitializing LLM Story Context Generator (this may take a while for model loading)...")
    story_context_generator = StoryContextGenerator(chromadb_manager=chroma_manager)
    
    print("\n--- System Initialization Complete ---")
    return chroma_manager, story_context_generator

def main():
    chroma_manager, story_generator = initialize_system()

    while True:
        print("\n==================================================")
        story_details_prompt = input("ป้อนรายละเอียดเนื้อเรื่อง (หรือพิมพ์ 'exit' เพื่อจบ): ")
        print("==================================================")

        if not story_details_prompt.strip():
            print("ไม่ได้ป้อนรายละเอียดเนื้อเรื่อง ลองใหม่อีกครั้ง")
            continue
        
        if story_details_prompt.lower() == 'exit':
            print("กำลังออกจากโปรแกรม...")
            break

        print(f"\nกำลังประมวลผลเนื้อเรื่อง: '{story_details_prompt}'...")
        print("AI (LLM) กำลังตัดสินใจว่าจะค้นหาข้อมูลอะไรจากฐานข้อมูล...")

        # LLM-driven RAG using StoryContextGenerator
        final_output_from_llm = story_generator.generate_story_elements(story_details_prompt)
        
        # Ensure the output structure is as expected by schema/place.and.characters.spec.schema.json
        # The LLM's output (final_output_from_llm) should already be in the correct format
        # based on how _execute_tool_call accumulates results.
        # We just need to make sure the top-level keys are present.

        output_characters = []
        # The LLM results are already lists of full dicts from ChromaDB metadata
        if final_output_from_llm.get("Characters"):
            for char_meta in final_output_from_llm["Characters"]:
                output_characters.append({
                    "character_id": char_meta.get("character_id", "N/A"),
                    "character_name": char_meta.get("character_name", "N/A"),
                    "character_profession": char_meta.get("character_profession", "N/A"),
                    "bio_detail": char_meta.get("bio_detail", "N/A")
                    # "similarity_score": char_meta.get("similarity_score") # Can be included if needed
                })

        output_locations = []
        if final_output_from_llm.get("Locations"):
            for loc_meta in final_output_from_llm["Locations"]:
                output_locations.append({
                    "location_id": loc_meta.get("location_id", "N/A"),
                    "location_name": loc_meta.get("location_name", "N/A"),
                    "location_type": loc_meta.get("location_type", "N/A"),
                    "location_detail": loc_meta.get("location_detail", "N/A")
                    # "similarity_score": loc_meta.get("similarity_score") # Can be included if needed
                })
            
        structured_output_for_display = {
            "Characters": output_characters,
            "Locations": output_locations
        }
        
        print("\n--- ผลลัพธ์จาก AI (รูปแบบ JSON) ---")
        print(json.dumps(structured_output_for_display, indent=2, ensure_ascii=False))
        print("----------------------------------")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"เกิดข้อผิดพลาดร้ายแรงในโปรแกรมหลัก: {e}")
        import traceback
        traceback.print_exc()
