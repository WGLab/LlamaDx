import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.memory_insert_params import Document
from typing import List

# Embedding configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE_TOKENS = 512
OVERLAP_SIZE_TOKENS = 10

class LlamaDxRAG:
    def __init__(self, docs_dir: str, chroma_dir: str, memory_bank_id: str):
        self.docs_dir = docs_dir
        self.memory_bank_id = memory_bank_id
        self.chroma_dir = chroma_dir
        self.initialize_agent()


    def initialize_agent(self):
        # Let's see what providers are available
        # Providers determine where and how your data is stored
        self.chroma_client = chromadb.PersistentClient(
                settings=Settings(
                    persist_directory=self.chroma_dir
                )
            )
        # Ensure collection exists
        collections = self.chroma_client.list_collections()
        if any(col.name == self.memory_bank_id for col in collections):
            print(f"The collection '{self.memory_bank_id}' already exists.")
        else:
            print(f"The collection '{self.memory_bank_id}' does not exist. Creating and initializing...")
            
            # Create collection with embedding model
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
            collections = self.chroma_client.create_collection(
                name=self.memory_bank_id,
                embedding_function=embedding_function
            )
            self.insert_documents(collections)
    # Text chunking function
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Splits a text into overlapping chunks.
        
        Args:
            text (str): The input text to split.
            chunk_size (int): Maximum number of tokens in a chunk.
            overlap (int): Number of overlapping tokens between consecutive chunks.

        Returns:
            List[str]: List of text chunks.
        """
        words = text.split()  # Split by whitespace
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def insert_documents(self, collections):
        # Load and process documents
        documents = []
        metadatas = []
        ids = []

        for filename in os.listdir(self.docs_dir):
            if filename.endswith((".txt", ".md")):
                file_path = os.path.join(self.docs_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    chunks = self.chunk_text(content, CHUNK_SIZE_TOKENS, OVERLAP_SIZE_TOKENS)
                    for idx, chunk in enumerate(chunks):
                        chunk_id = f"{filename}_chunk_{idx}"
                        documents.append(chunk)
                        metadatas.append({"filename": filename, "chunk": idx})
                        ids.append(chunk_id)

        # Insert documents into collection
        if documents:
            collections.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def generate_documents(self, query: str, top_k: int):
        """Helper function to print query results in a readable format

        Args:
            query (str): The search query to execute
        """
        # print(f"\nQuery: {query}")
        # print("-" * 50)
        collections = self.chroma_client.get_collection(name=self.memory_bank_id)
        results = collections.query(
            query_texts=[query],
            n_results=top_k  # Retrieve top k results
        )
        curated_text = ""
        # Iterate over results
        for query_idx, (docs, distances) in enumerate(zip(results["documents"], results["distances"])):
            #print(f"\nResults for Query {query_idx + 1}")
            for i, (doc, score) in enumerate(zip(docs, distances)):
                curated_text += f"Document {i+1} (Score: {score:.3f})\n"
                curated_text += "=" * 40 + "\n"
                curated_text += doc + '\n'
                curated_text += "=" * 40 + "\n"
        return curated_text

# def main():
#     docs_dir = './rag_genetics_small'
#     chroma_dir = "./chroma"  # Directory to persist Chroma data
#     llama_rag = LlamaDxRAG(docs_dir=docs_dir, chroma_dir = chroma_dir)
#     query = 'Cystic Fibrosis'
#     llama_rag_genetics = llama_rag.generate_documents(query, top_k = 5)
