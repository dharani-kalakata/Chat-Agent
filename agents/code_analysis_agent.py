from typing import Dict, Any, AsyncIterator, Optional
import ast
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .base_agent import BaseAgent
from services.langchain_service import LangChainService
from config import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL

class CodeAnalysisAgent(BaseAgent):
    """Agent specialized for code analysis and insight generation"""

    def __init__(self, langchain_service: LangChainService = None):
        self.langchain_service = langchain_service or LangChainService()
        self.analysis_template = PromptTemplate(
            input_variables=["code", "file_path", "analysis_type"],
            template="""Analyze the following Python code and provide insights:

File: {file_path}
Analysis Type: {analysis_type}

```python
{code}
```

Provide a detailed analysis focusing on:
{analysis_type}

Your analysis should include:
- Summary of code structure and functionality
- Key components and their interactions
- Areas that would benefit from test coverage
- Potential edge cases to consider
- Any code quality issues or improvements

Analysis:"""
        )

    def _extract_code_structure(self, code: str) -> Dict[str, Any]:
        """Extract basic structure from code using AST parsing"""
        try:
            tree = ast.parse(code)
            functions, classes, imports = [], [], []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({"name": node.name, "line_number": node.lineno})
                elif isinstance(node, ast.ClassDef):
                    methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
                    classes.append({"name": node.name, "methods": methods, "line_number": node.lineno})
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(node.module if hasattr(node, "module") else node.names[0].name)
            return {"functions": functions, "classes": classes, "imports": imports}
        except SyntaxError:
            return {"error": "Syntax error in code"}

    async def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run code analysis task and return formatted analysis"""
        file_path, code, analysis_type = input_data.get("file_path", ""), input_data.get("code", ""), input_data.get("analysis_type", "general")
        if not code and file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        if not code:
            return {"error": "No code provided for analysis"}
        model = input_data.get("model", DEFAULT_OPENAI_MODEL if input_data.get("provider") == "openai" else DEFAULT_OLLAMA_MODEL)
        prompt = self.analysis_template.format(code=code, file_path=file_path, analysis_type=analysis_type)
        response = await self.langchain_service.generate(prompt=prompt, model=model, provider=input_data.get("provider"), **kwargs)
        return {"file_path": file_path, "analysis": response["response"], "code_structure": self._extract_code_structure(code)}
    
    async def stream_run(self, input_data: Dict[str, Any], **kwargs) -> AsyncIterator[str]:
        """Stream code analysis as it's generated"""
        file_path, code, analysis_type = input_data.get("file_path", ""), input_data.get("code", ""), input_data.get("analysis_type", "general")
        if not code and file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except Exception as e:
                yield f"Error reading file: {str(e)}"
                return
        
        if not code:
            yield "Error: No code provided for analysis"
            return
        
        model = input_data.get("model", DEFAULT_OPENAI_MODEL if input_data.get("provider") == "openai" else DEFAULT_OLLAMA_MODEL)
        prompt = self.analysis_template.format(code=code, file_path=file_path, analysis_type=analysis_type)
        
        async for chunk in self.langchain_service.stream_generate(
            prompt=prompt,
            model=model,
            provider=input_data.get("provider"),
            **kwargs
        ):
            yield chunk
