import os
from dotenv import load_dotenv
from .chromadb_manager import ChromaDBManager, CHROMA_DB_PATH

# สร้าง instance ของ ChromaDBManager
db_manager = ChromaDBManager(path=CHROMA_DB_PATH)

def search_characters_hybrid(query_text, n_results=3):
    """ค้นหาตัวละครแบบผสมระหว่าง keyword matching และ semantic search"""
    # Step 1: แยกคำสำคัญจากคำค้นหา
    keywords = ["intern", "director", "manager", "executive", "coordinator", "technician"]
    matching_keywords = [k for k in keywords if k.lower() in query_text.lower()]
    
    # Step 2: ค้นหาตัวละครที่มีตำแหน่งงานตรงกับคำค้นหาก่อน (exact matching)
    exact_matches = []
    
    # ดึงตัวละครทั้งหมดจาก collection
    all_characters = db_manager.character_collection.get(
        include=["metadatas"])
    
    if all_characters and "metadatas" in all_characters and all_characters["metadatas"]:
        for metadata in all_characters["metadatas"]:
            if "character_profession" not in metadata:
                continue
                
            profession = metadata["character_profession"].lower()
            for keyword in matching_keywords:
                if keyword.lower() in profession:
                    exact_matches.append(metadata)
                    break
    
    # Step 3: ถ้าไม่พบการตรงกันหรือต้องการผลลัพธ์เพิ่มเติม ใช้ semantic search
    if len(exact_matches) < n_results:
        # จำนวนผลลัพธ์ที่ต้องการเพิ่มเติม
        remaining_count = n_results - len(exact_matches)
        
        # ใช้ semantic search สำหรับผลลัพธ์เพิ่มเติม
        semantic_results = db_manager.query_characters(query_text, n_results=remaining_count+len(exact_matches))
        
        # กรองผลลัพธ์ที่ซ้ำกัน
        semantic_filtered = []
        for result in semantic_results:
            if not any(result.get('character_id') == m.get('character_id') for m in exact_matches):
                semantic_filtered.append(result)
                if len(semantic_filtered) >= remaining_count:
                    break
        
        # รวมผลลัพธ์
        combined_results = exact_matches + semantic_filtered
        return combined_results[:n_results]
    
    return exact_matches[:n_results]

def search_locations_hybrid(query_text, n_results=3):
    """ค้นหาสถานที่แบบผสม - ตอนนี้เริ่มต้นด้วย semantic search"""
    return db_manager.query_locations(query_text, n_results)