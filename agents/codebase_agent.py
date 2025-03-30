import os
import glob
import json
from typing import Dict, Any, AsyncIterator, List, Optional, Union
import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableMap

from .base_agent import BaseAgent
from services.langchain_service import LangChainService
from config import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodebaseAgent(BaseAgent):
    """Agent for analyzing codebases, answering questions, and generating tests."""

    def __init__(self, langchain_service: LangChainService = None):
        self.langchain_service = langchain_service or LangChainService()
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Vector store path
        self.persist_directory = "chroma_db"
        self.vector_store = None
        self.code_documents = []
        
        # Prompt templates
        self.code_qa_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful programming assistant. Use the following code context to answer the question.

CODE CONTEXT:
{context}

QUESTION:
{question}

When answering:
1. Be specific and refer to the code when possible
2. If the answer is not in the context, say so instead of making things up
3. If showing code examples, use proper formatting
4. Focus on explaining how the code works

ANSWER:"""
        )
        
        self.test_generation_template = PromptTemplate(
            input_variables=["code", "file_path"],
            template="""Generate comprehensive pytest test cases for the following Python code:

FILE PATH: {file_path}

CODE:
```python
{code}
```

Create pytest test cases that:
1. Cover all public methods and functions
2. Include positive test cases for expected behavior
3. Include negative test cases for edge cases and error handling
4. Use appropriate pytest features (fixtures, parametrization, etc.)
5. Include docstrings explaining the purpose of each test

Format the tests as a complete pytest test file that can be saved and executed.

TEST CODE:"""
        )

    def _load_single_file(self, file_path: str) -> List[Document]:
        """Load and process a single Python file."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create a Document object
            return [Document(page_content=content, metadata={"source": file_path})]
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []
            
    def _load_directory(self, directory_path: str, file_extension: str = "*.py") -> List[Document]:
        """Load all Python files from a directory."""
        documents = []
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return []
            
        # Find all Python files in the directory and subdirectories
        pattern = os.path.join(directory_path, "**", file_extension)
        file_paths = glob.glob(pattern, recursive=True)
        
        for file_path in file_paths:
            documents.extend(self._load_single_file(file_path))
        
        return documents
    
    def _split_code_documents(self, documents: List[Document]) -> List[Document]:
        """Split code documents into smaller chunks using appropriate splitter."""
        if not documents:
            return []
            
        # Create code-specific text splitter
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        return splitter.split_documents(documents)
        
    def load_codebase(self, path: str, file_extension: str = "*.py") -> Dict[str, Any]:
        """Load code from a file or directory into the vector store."""
        try:
            if os.path.isfile(path):
                documents = self._load_single_file(path)
                logger.info(f"Loaded 1 file from: {path}")
            elif os.path.isdir(path):
                documents = self._load_directory(path, file_extension)
                logger.info(f"Loaded {len(documents)} files from: {path}")
            else:
                return {"error": f"Path does not exist: {path}", "success": False}
                
            if not documents:
                return {"error": "No documents loaded", "success": False}
                
            # Process and split documents
            self.code_documents = documents
            chunks = self._split_code_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks for embedding")
            
            # Create or update vector store
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding_function=self.embeddings,  # Updated parameter name
                persist_directory=self.persist_directory
            )
            
            return {
                "success": True,
                "message": f"Loaded {len(documents)} files with {len(chunks)} chunks",
                "file_count": len(documents),
                "chunk_count": len(chunks)
            }
        except Exception as e:
            logger.error(f"Error loading codebase: {e}")
            return {"error": str(e), "success": False}
    
    async def answer_question(self, 
                             question: str,
                             model: str,
                             provider: str = "ollama", 
                             **kwargs) -> Dict[str, Any]:
        """Answer questions about the loaded codebase using RAG."""
        if not self.vector_store:
            return {"error": "No codebase loaded. Please load a codebase first.", "success": False}
            
        try:
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Get relevant documents - use the string directly to avoid dict errors
            try:
                # First attempt using get_relevant_documents to avoid the error
                retrieved_docs = retriever.get_relevant_documents(question)
                logger.info(f"Retrieved {len(retrieved_docs)} documents using get_relevant_documents")
            except Exception as retriever_error:
                logger.warning(f"Error using get_relevant_documents: {retriever_error}")
                # Fallback to invoke with proper unpacking if needed
                try:
                    # Some retrievers expect a dict with "query" key
                    retrieval_result = retriever.invoke({"query": question})
                    if isinstance(retrieval_result, list):
                        retrieved_docs = retrieval_result
                    elif isinstance(retrieval_result, dict) and "documents" in retrieval_result:
                        retrieved_docs = retrieval_result["documents"]
                    else:
                        raise ValueError(f"Unexpected retrieval result format: {type(retrieval_result)}")
                except Exception as invoke_error:
                    logger.error(f"Both retrieval methods failed: {invoke_error}")
                    raise
            
            # Process document contents safely with robust type checking
            context_parts = []
            sources = []
            
            for doc in retrieved_docs:
                # Safely get page_content
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    # Ensure content is a string
                    if not isinstance(content, str):
                        logger.warning(f"Document content is not a string: {type(content)}. Converting to string.")
                        content = str(content)
                    context_parts.append(content)
                else:
                    logger.warning(f"Document has no page_content attribute: {type(doc)}")
                
                # Safely get source
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                    source = doc.metadata.get("source", "Unknown source")
                    sources.append(source)
                else:
                    logger.warning(f"Document has no metadata dictionary: {type(getattr(doc, 'metadata', None))}")
                    sources.append("Unknown source")
            
            # Join the context parts
            context = "\n\n".join(context_parts)
            
            # Format prompt with context and question
            prompt = self.code_qa_template.format(context=context, question=question)
            
            # Generate answer with LLM
            response = await self.langchain_service.generate(
                prompt=prompt,
                model=model,
                provider=provider,
                **kwargs
            )
            
            return {
                "question": question,
                "answer": response["response"],
                "sources": list(set(sources)),
                "success": True
            }
        except Exception as e:
            logger.error(f"Error answering question: {e}", exc_info=True)
            return {"error": str(e), "success": False}
            
    async def stream_answer_question(self, 
                                   question: str,
                                   model: str,
                                   provider: str = "ollama",
                                   **kwargs) -> AsyncIterator[str]:
        """Stream answers to questions about the loaded codebase."""
        if not self.vector_store:
            yield json.dumps({"error": "No codebase loaded. Please load a codebase first."})
            return
            
        try:
            # Create retriever and get context
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Get relevant documents - use the string directly to avoid dict errors
            try:
                # First attempt using get_relevant_documents to avoid the error
                retrieved_docs = retriever.get_relevant_documents(question)
                logger.info(f"Retrieved {len(retrieved_docs)} documents using get_relevant_documents")
            except Exception as retriever_error:
                logger.warning(f"Error using get_relevant_documents: {retriever_error}")
                # Fallback to invoke with proper unpacking if needed
                try:
                    # Some retrievers expect a dict with "query" key
                    retrieval_result = retriever.invoke({"query": question})
                    if isinstance(retrieval_result, list):
                        retrieved_docs = retrieval_result
                    elif isinstance(retrieval_result, dict) and "documents" in retrieval_result:
                        retrieved_docs = retrieval_result["documents"]
                    else:
                        raise ValueError(f"Unexpected retrieval result format: {type(retrieval_result)}")
                except Exception as invoke_error:
                    logger.error(f"Both retrieval methods failed: {invoke_error}")
                    raise
            
            # Process document contents safely with robust type checking
            context_parts = []
            
            for doc in retrieved_docs:
                # Safely get page_content
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    # Ensure content is a string
                    if not isinstance(content, str):
                        logger.warning(f"Document content is not a string: {type(content)}. Converting to string.")
                        content = str(content)
                    context_parts.append(content)
                else:
                    logger.warning(f"Document has no page_content attribute: {type(doc)}")
            
            # Join the context parts
            context = "\n\n".join(context_parts)
            
            # Format prompt with context and question
            prompt = self.code_qa_template.format(context=context, question=question)
            
            # Stream response
            async for chunk in self.langchain_service.stream_generate(
                prompt=prompt,
                model=model,
                provider=provider,
                **kwargs
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming answer: {e}", exc_info=True)
            yield json.dumps({"error": str(e)})

    async def generate_tests(self, 
                           file_path: str, 
                           model: str,
                           provider: str = "ollama",
                           **kwargs) -> Dict[str, Any]:
        """Generate test cases for a specific file."""
        try:
            # Load file content
            documents = self._load_single_file(file_path)
            if not documents:
                return {"error": f"Could not load file: {file_path}", "success": False}
                
            code = documents[0].page_content
            
            # Format prompt
            prompt = self.test_generation_template.format(
                code=code,
                file_path=file_path
            )
            
            # Generate test cases with LLM
            response = await self.langchain_service.generate(
                prompt=prompt,
                model=model,
                provider=provider,
                **kwargs
            )
            
            # Extract filename for test file naming
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            test_file_name = f"test_{base_name}.py"
            
            return {
                "file_path": file_path,
                "test_code": response["response"],
                "suggested_test_filename": test_file_name,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return {"error": str(e), "success": False}
            
    async def stream_generate_tests(self,
                                  file_path: str,
                                  model: str,
                                  provider: str = "ollama",
                                  **kwargs) -> AsyncIterator[str]:
        """Stream test case generation for a specific file."""
        try:
            # Load file content
            documents = self._load_single_file(file_path)
            if not documents:
                yield json.dumps({"error": f"Could not load file: {file_path}"})
                return
                
            code = documents[0].page_content
            
            # Format prompt
            prompt = self.test_generation_template.format(
                code=code,
                file_path=file_path
            )
            
            # Stream test generation
            async for chunk in self.langchain_service.stream_generate(
                prompt=prompt,
                model=model,
                provider=provider,
                **kwargs
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming test generation: {e}")
            yield json.dumps({"error": str(e)})

    async def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run agent task based on the operation type specified in input_data."""
        operation = input_data.get("operation")
        
        if operation == "load_codebase":
            return self.load_codebase(
                path=input_data.get("path", ""),
                file_extension=input_data.get("file_extension", "*.py")
            )
        elif operation == "answer_question":
            return await self.answer_question(
                question=input_data.get("question", ""),
                model=input_data.get("model"),
                provider=input_data.get("provider", "ollama"),
                **kwargs
            )
        elif operation == "generate_tests":
            return await self.generate_tests(
                file_path=input_data.get("file_path", ""),
                model=input_data.get("model"),
                provider=input_data.get("provider", "ollama"),
                **kwargs
            )
        else:
            return {"error": f"Unknown operation: {operation}", "success": False}
            
    async def stream_run(self, input_data: Dict[str, Any], **kwargs) -> AsyncIterator[str]:
        """Stream agent task based on the operation type specified in input_data."""
        operation = input_data.get("operation")
        
        if operation == "answer_question":
            async for chunk in self.stream_answer_question(
                question=input_data.get("question", ""),
                model=input_data.get("model"),
                provider=input_data.get("provider", "ollama"),
                **kwargs
            ):
                yield chunk
        elif operation == "generate_tests":
            async for chunk in self.stream_generate_tests(
                file_path=input_data.get("file_path", ""),
                model=input_data.get("model"),
                provider=input_data.get("provider", "ollama"),
                **kwargs
            ):
                yield chunk
        else:
            yield json.dumps({"error": f"Unknown streaming operation: {operation}"})
    
    def get_retriever(self, k: int = 5, search_type: str = "similarity"):
        """Expose the retriever for debugging purposes."""
        if not self.vector_store:
            raise ValueError("No vector store available. Load a codebase first.")
        
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
