# src/chromadb_manager.py
import chromadb
import os
import json
from dotenv import load_dotenv
from .embedding_utils import get_embedding_function_for_chroma, get_embeddings # get_embeddings for manual query embedding

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data_store")
CHARACTER_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_CHARACTERS", "characters_collection")
LOCATION_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_LOCATIONS", "locations_collection")

class ChromaDBManager:
    def __init__(self, path: str = CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        # Use the SentenceTransformerEmbeddingFunction directly provided by ChromaDB for collections
        self.chroma_embedding_function = get_embedding_function_for_chroma()
        
        self.character_collection = self._get_or_create_collection(
            CHARACTER_COLLECTION_NAME,
            self.chroma_embedding_function
        )
        self.location_collection = self._get_or_create_collection(
            LOCATION_COLLECTION_NAME,
            self.chroma_embedding_function
        )
        print(f"ChromaDB client initialized. Data path: {path}")
        print(f"Character collection: '{CHARACTER_COLLECTION_NAME}', Item count: {self.character_collection.count()}")
        print(f"Location collection: '{LOCATION_COLLECTION_NAME}', Item count: {self.location_collection.count()}")

    def _get_or_create_collection(self, collection_name: str, embedding_function):
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_function 
            )
        except Exception:
             print(f"Collection '{collection_name}' not found or embedding function mismatch. Creating/getting with new EF...")
             collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"} 
            )
             print(f"Collection '{collection_name}' ensured.")
        return collection

    def _add_data_to_collection(self, collection, items_data: list[dict], id_key: str, document_key: str, type_name: str):
        if not items_data:
            return

        ids = [item[id_key] for item in items_data]
        documents = [item[document_key] for item in items_data]
        # Metadatas are all fields of the item for easy retrieval
        metadatas = [dict(item) for item in items_data]

        existing_ids_results = collection.get(ids=ids)
        existing_ids = set(existing_ids_results['ids'])
        
        new_ids = []
        new_documents = []
        new_metadatas = []

        for i, item_id in enumerate(ids):
            if item_id not in existing_ids:
                new_ids.append(item_id)
                new_documents.append(documents[i])
                new_metadatas.append(metadatas[i])

        if new_ids:
            collection.add(
                ids=new_ids,
                documents=new_documents, # Chroma uses these documents with its EF
                metadatas=new_metadatas
            )
            print(f"Added {len(new_ids)} new {type_name} to '{collection.name}'.")
        else:
            print(f"No new {type_name} to add to '{collection.name}'. All provided IDs already exist.")

    def add_characters(self, characters_data: list[dict]):
        self._add_data_to_collection(self.character_collection, characters_data, "character_id", "bio_detail", "characters")

    def add_locations(self, locations_data: list[dict]):
        self._add_data_to_collection(self.location_collection, locations_data, "location_id", "location_detail", "locations")

    def query_collection(self, collection_name: str, query_text: str, n_results: int = 3) -> list[dict]:
        if collection_name == self.character_collection.name:
            collection_to_query = self.character_collection
        elif collection_name == self.location_collection.name:
            collection_to_query = self.location_collection
        else:
            print(f"Error: Collection '{collection_name}' not recognized.")
            return []

        if not query_text:
            return []

        # Generate query embedding using the same model defined in embedding_utils
        # This is for querying, not for adding to collection (collection uses its own EF)
        query_embedding_list = get_embeddings([query_text])
        if not query_embedding_list:
            print("Error: Could not generate query embedding.")
            return []
        
        query_embedding = query_embedding_list[0]

        results = collection_to_query.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['metadatas', 'distances']
        )
        
        retrieved_items = []
        if results and results.get('metadatas') and results['metadatas'][0]:
            for i, meta in enumerate(results['metadatas'][0]):
                item = dict(meta) 
                # Cosine distance to similarity: similarity = (1 + (1 - distance)) / 2 if distance is -1 to 1
                # Or simply 1 - distance if distance is 0 to 2 (like from SentenceTransformers)
                # ChromaDB's cosine distance is typically 1 - cosine_similarity, so score is 1 - distance.
                # For hnsw:space=cosine, distance is 1 - cosine_similarity.
                item['similarity_score'] = 1 - results['distances'][0][i] 
                retrieved_items.append(item)
        return retrieved_items

    def query_characters(self, query_text: str, n_results: int = 2) -> list[dict]:
        return self.query_collection(self.character_collection.name, query_text, n_results)

    def query_locations(self, query_text: str, n_results: int = 2) -> list[dict]:
        return self.query_collection(self.location_collection.name, query_text, n_results)

    def load_data_from_json(self, json_file_path: str = "data/sample_data.json"):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "characters" in data:
                self.add_characters(data["characters"])
            if "locations" in data:
                self.add_locations(data["locations"])
            print(f"Data loading process from '{json_file_path}' complete.")
        except FileNotFoundError:
            print(f"Error: Data file '{json_file_path}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{json_file_path}'.")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")


if __name__ == '__main__':
    manager = ChromaDBManager()
    manager.load_data_from_json()

    print("\n--- Testing Character Query ---")
    char_results = manager.query_characters("นักสำรวจผู้กล้าหาญ", n_results=1)
    if char_results:
        print(json.dumps(char_results, indent=2, ensure_ascii=False))
    else:
        print("  No characters found.")

    print("\n--- Testing Location Query ---")
    loc_results = manager.query_locations("ป่าลึกลับ", n_results=1)
    if loc_results:
        print(json.dumps(loc_results, indent=2, ensure_ascii=False))
    else:
        print("  No locations found.")
