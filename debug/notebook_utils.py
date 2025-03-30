from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
from typing import List, Dict
import os

class DebugUtils:
    """Utilities for debugging vector store retrieval in Jupyter notebooks."""
    
    def __init__(self, chroma_db_path: str = "chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        if os.path.exists(chroma_db_path):
            self.vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=self.embeddings)
        else:
            raise ValueError(f"Chroma DB not found at {chroma_db_path}. Please load a codebase first.")
    
    def visualize_retrieval(self, question: str, k: int = 5, search_type: str = "similarity") -> None:
        """Visualize retrieved chunks and their similarity scores."""
        retriever = self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
        retrieved_docs = retriever.get_relevant_documents(question)
        
        # Display question
        display(Markdown(f"### Question: {question}"))
        
        # Display retrieved chunks
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", "Unknown Source")
            content = doc.page_content
            score = doc.metadata.get("score", "N/A")
            
            display(Markdown(f"#### Chunk {i + 1}"))
            display(Markdown(f"**Source:** {source}"))
            display(Markdown(f"**Score:** {score}"))
            display(Markdown(f"```python\n{content}\n```"))
    
    def plot_similarity_scores(self, question: str, k: int = 5, search_type: str = "similarity") -> None:
        """Plot similarity scores of retrieved chunks."""
        retriever = self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
        retrieved_docs = retriever.get_relevant_documents(question)
        
        scores = [doc.metadata.get("score", 0) for doc in retrieved_docs]
        sources = [doc.metadata.get("source", "Unknown") for doc in retrieved_docs]
        
        plt.figure(figsize=(10, 6))
        plt.barh(sources, scores, color="skyblue")
        plt.xlabel("Similarity Score")
        plt.ylabel("Source")
        plt.title("Similarity Scores of Retrieved Chunks")
        plt.gca().invert_yaxis()
        plt.show()
