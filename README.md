# ChromaDB RAG Project for Story Character and Location Generation

This project demonstrates how to build a Retrieval Augmented Generation (RAG) system using ChromaDB as a vector database, `jina-embeddings-v2-base-en` for text embeddings, and `CohereLabs/c4ai-command-r-v01` (via Hugging Face Transformers) as the Large Language Model (LLM). The LLM uses a specific "Rendered Single-Step Tool Use Prompt" format to decide which tools to call (database searches) for finding relevant characters and locations based on a user's story prompt.

## Features

-   **Vector Database:** ChromaDB for storing and querying character/location embeddings.
-   **Embeddings:** `jina-embeddings-v2-base-en` for text embeddings.
-   **RAG with LLM Tool Use:** Integrates Cohere Command R model, which interprets story prompts and decides which database search tools to use via a precisely formatted tool-calling mechanism.
-   **Customizable Data:** Easy addition of character/location data via `data/sample_data.json`.
-   **Structured Output:** Generates output in a predefined JSON schema.
-   **Environment Management:** `environment.yaml` for Conda and `requirements.txt` for pip.
-   **Configurable:** `.env` file for model names, paths, and API tokens.

## Prerequisites

-   Conda (recommended) or Python **3.12** with pip.
-   NVIDIA GPU with appropriate drivers. The project is configured to use PyTorch with **CUDA 12.1**, so your drivers must support this CUDA version or newer.
-   Sufficient VRAM (RTX 4090 as specified by user is ample).
-   Hugging Face Hub Token (if `CohereLabs/c4ai-command-r-v01` is gated or requires it). Set in `.env`.

## Project Structure
(โครงสร้างไฟล์เหมือนเดิม)

## Setup

1.  **Clone/Create Files:** Set up the project directory and files as listed.

2.  **Environment Setup (Choose Conda or Pip/Venv):**

    **A. Conda Environment (Recommended):**
    ```bash
    conda env create -f environment.yaml
    conda activate Lifeforge
    ```

    **B. Pip with Virtual Environment:**
    If you prefer using pip and a virtual environment:
    ```bash
    # Create a virtual environment (e.g., named .venv)
    python -m venv .venv

    # Activate the virtual environment
    # On Windows:
    # .venv\Scripts\activate
    # On macOS/Linux:
    # source .venv/bin/activate

    # IMPORTANT: Install PyTorch with CUDA 12.1 support first
    # Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for the latest command if needed
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

    # Install other dependencies
    pip install -r requirements.txt
    ```

3.  **Environment Variables (`.env`):**
    Create `.env` in the project root. Example:
    ```ini
    EMBEDDING_MODEL_NAME="jinaai/jina-embeddings-v2-base-en"
    LLM_MODEL_NAME="CohereLabs/c4ai-command-r-v01"
    CHROMA_DB_PATH="./chroma_data_store"
    CHROMA_COLLECTION_CHARACTERS="characters_collection"
    CHROMA_COLLECTION_LOCATIONS="locations_collection"
    # HUGGING_FACE_HUB_TOKEN="your_hf_token_here" # IMPORTANT if model is gated
    ```

4.  **Hugging Face Login (if needed):**
    If the LLM model requires authentication:
    ```bash
    huggingface-cli login
    ```

## Running the Application

From the project's root directory, after activating your chosen environment (`Lifeforge` for Conda, or your venv for pip):

```bash
python -m src.main
