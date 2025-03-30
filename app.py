from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Literal

from services.langchain_service import LangChainService
from agents.qa_agent import QAAgent
from agents.code_analysis_agent import CodeAnalysisAgent
from utils.response_formatter import format_response
from config import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL

# Initialize FastAPI application
app = FastAPI(title="LLM API", description="API for LLM integration with Ollama and OpenAI using LangChain")

# Initialize services
langchain_service = LangChainService()
qa_agent = QAAgent(langchain_service)
code_analysis_agent = CodeAnalysisAgent(langchain_service)

# Request data models
class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = DEFAULT_OLLAMA_MODEL
    stream: bool = False
    provider: Optional[Literal["ollama", "openai"]] = "ollama"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class QARequest(BaseModel):
    question: str
    model: Optional[str] = None
    stream: bool = False
    provider: Optional[Literal["ollama", "openai"]] = "ollama"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class CodeAnalysisRequest(BaseModel):
    file_path: Optional[str] = ""
    code: Optional[str] = ""
    analysis_type: Optional[str] = "general"
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

@app.post("/qa")
async def qa_endpoint(request: QARequest):
    """Question answering endpoint with specialized formatting"""
    kwargs = {"temperature": request.temperature}
    if request.max_tokens:
        kwargs["max_tokens"] = request.max_tokens
    
    try:
        input_data = {
            "question": request.question,
            "provider": request.provider,
            "model": request.model,
        }
        
        if request.stream:
            return StreamingResponse(
                qa_agent.stream_run(input_data, **kwargs),
                media_type="text/plain"
            )
        else:
            return await qa_agent.run(input_data, **kwargs)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-code")
async def analyze_code(request: CodeAnalysisRequest):
    """Analyze code and provide insights"""
    kwargs = {"temperature": request.temperature}
    if request.max_tokens:
        kwargs["max_tokens"] = request.max_tokens
    try:
        input_data = {
            "file_path": request.file_path,
            "code": request.code,
            "analysis_type": request.analysis_type,
            "provider": request.provider,
            "model": request.model,
        }
        if request.stream:
            return StreamingResponse(
                code_analysis_agent.stream_run(input_data, **kwargs),
                media_type="text/plain"
            )
        else:
            analysis_result = await code_analysis_agent.run(input_data, **kwargs)
            # Format the response to standardize API output
            formatted_response = {
                "response": analysis_result["analysis"],
                "file_path": analysis_result["file_path"],
                "code_structure": analysis_result["code_structure"]
            }
            return formatted_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Retrieve available models from all configured providers"""
    try:
        return langchain_service.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))