from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Literal, List

from services.langchain_service import LangChainService
from agents.codebase_agent import CodebaseAgent
from utils.response_formatter import format_response
from config import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL

# Initialize FastAPI application
app = FastAPI(title="LLM API", description="API for LLM integration with Ollama and OpenAI using LangChain")

# Initialize services
langchain_service = LangChainService()
codebase_agent = CodebaseAgent(langchain_service)

# Request data models
class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = DEFAULT_OLLAMA_MODEL
    stream: bool = False
    provider: Optional[Literal["ollama", "openai"]] = "ollama"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class CodebaseLoadRequest(BaseModel):
    path: str
    file_extension: Optional[str] = "*.py"

class CodebaseQuestionRequest(BaseModel):
    question: str
    model: Optional[str] = None
    stream: bool = False
    provider: Optional[Literal["ollama", "openai"]] = "ollama"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class GenerateTestsRequest(BaseModel):
    file_path: str
    model: Optional[str] = None
    stream: bool = False
    provider: Optional[Literal["ollama", "openai"]] = "ollama"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text using LLM with optional streaming support"""
    kwargs = {"temperature": request.temperature}
    if request.max_tokens:
        kwargs["max_tokens"] = request.max_tokens
    
    try:
        # Select appropriate model based on provider
        model = DEFAULT_OPENAI_MODEL if request.provider == "openai" and (not request.model or request.model == DEFAULT_OLLAMA_MODEL) else request.model
            
        if request.stream:
            return StreamingResponse(
                langchain_service.get_formatted_response_streaming(
                    prompt=request.prompt,
                    model=model,
                    provider=request.provider,
                    **kwargs
                ),
                media_type="text/plain"
            )
        else:
            response = await langchain_service.get_formatted_response_non_streaming(
                prompt=request.prompt,
                model=model,
                provider=request.provider,
                **kwargs
            )
            return await format_response(response)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Retrieve available models from all configured providers"""
    try:
        return langchain_service.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New endpoints for codebase analysis

@app.post("/codebase/load")
async def load_codebase(request: CodebaseLoadRequest):
    """Load and index a codebase for analysis"""
    try:
        input_data = {
            "operation": "load_codebase",
            "path": request.path,
            "file_extension": request.file_extension
        }
        
        result = await codebase_agent.run(input_data)
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Unknown error loading codebase"))
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/codebase/question")
async def codebase_question(request: CodebaseQuestionRequest):
    """Answer questions about the loaded codebase"""
    kwargs = {"temperature": request.temperature}
    if request.max_tokens:
        kwargs["max_tokens"] = request.max_tokens
        
    try:
        # Select appropriate model based on provider
        model = DEFAULT_OPENAI_MODEL if request.provider == "openai" and not request.model else request.model or DEFAULT_OLLAMA_MODEL
        
        input_data = {
            "operation": "answer_question",
            "question": request.question,
            "model": model,
            "provider": request.provider
        }
        
        if request.stream:
            return StreamingResponse(
                codebase_agent.stream_run(input_data, **kwargs),
                media_type="text/plain"
            )
        else:
            result = await codebase_agent.run(input_data, **kwargs)
            if not result.get("success", False):
                raise HTTPException(status_code=400, detail=result.get("error", "Unknown error answering question"))
                
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/codebase/generate-tests")
async def generate_tests(request: GenerateTestsRequest):
    """Generate pytest test cases for a specified Python file"""
    kwargs = {"temperature": request.temperature}
    if request.max_tokens:
        kwargs["max_tokens"] = request.max_tokens
        
    try:
        # Select appropriate model based on provider
        model = DEFAULT_OPENAI_MODEL if request.provider == "openai" and not request.model else request.model or DEFAULT_OLLAMA_MODEL
        
        input_data = {
            "operation": "generate_tests",
            "file_path": request.file_path,
            "model": model,
            "provider": request.provider
        }
        
        if request.stream:
            return StreamingResponse(
                codebase_agent.stream_run(input_data, **kwargs),
                media_type="text/plain"
            )
        else:
            result = await codebase_agent.run(input_data, **kwargs)
            if not result.get("success", False):
                raise HTTPException(status_code=400, detail=result.get("error", "Unknown error generating tests"))
                
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthcheck", status_code=200)
async def healthcheck():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy", "api_version": "1.0.0"}