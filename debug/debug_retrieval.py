import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from rich.console import Console
from rich.syntax import Syntax
from typing import List, Dict

# Initialize console for pretty printing
console = Console()

# Path to the Chroma database
CHROMA_DB_PATH = "chroma_db"

def connect_to_vector_store() -> Chroma:
    """Connect to the Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    if not os.path.exists(CHROMA_DB_PATH):
        raise ValueError(f"Chroma DB not found at {CHROMA_DB_PATH}. Please load a codebase first.")
        
    return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

def debug_retrieval(question: str, k: int = 5, search_type: str = "similarity") -> None:
    """Debug the retrieval process by showing detailed information about retrieved chunks."""
    vector_store = connect_to_vector_store()
    retriever = vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
    
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(question)
    
    console.print(f"[bold green]Question:[/bold green] {question}")
    console.print(f"[bold blue]Top {k} Retrieved Chunks:[/bold blue]\n")
    
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "Unknown Source")
        content = doc.page_content
        score = doc.metadata.get("score", "N/A")
        
        console.print(f"[bold yellow]Chunk {i + 1}[/bold yellow]")
        console.print(f"[bold magenta]Source:[/bold magenta] {source}")
        console.print(f"[bold magenta]Score:[/bold magenta] {score}")
        
        # Display code with syntax highlighting
        syntax = Syntax(content, "python", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug vector store retrieval.")
    parser.add_argument("question", type=str, help="The question to debug.")
    parser.add_argument("--k", type=int, default=5, help="Number of top chunks to retrieve.")
    parser.add_argument("--search_type", type=str, default="similarity", help="Search type (e.g., similarity).")
    args = parser.parse_args()

    debug_retrieval(args.question, k=args.k, search_type=args.search_type)
