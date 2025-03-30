import requests
import json
import sys
from typing import Optional

def send_request(prompt, model=None, stream=False, provider="ollama", temperature=0.7, max_tokens=None):
    """Send a text generation request to the API"""
    url = "http://localhost:8000/generate"
    
    # Build request payload
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "stream": stream,
        "provider": provider
    }
    
    # Only include optional parameters when provided
    if model:
        data["model"] = model
    if temperature != 0.7:
        data["temperature"] = temperature
    if max_tokens:
        data["max_tokens"] = max_tokens

    print(f"Sending request to {url} with stream={stream}, provider={provider}, model={model or 'default'}")
    
    # Send request
    response = requests.post(url, headers=headers, json=data, stream=stream)

    # Process response
    if response.status_code == 200:
        if stream:
            # Handle streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')
                        try:
                            json_data = json.loads(line_text)
                            if "response" in json_data:
                                print(json_data["response"], end="", flush=True)
                            if json_data.get("done", False):
                                print()
                        except json.JSONDecodeError:
                            print(line_text, end="", flush=True)
                    except Exception as e:
                        print(f"Error processing streaming response: {e}")
        else:
            # Handle complete response
            try:
                response_json = response.json()
                if isinstance(response_json, dict) and "response" in response_json:
                    print(response_json["response"])
                else:
                    print(response_json)
            except json.JSONDecodeError:
                print(response.text)
    else:
        print(f"Error: {response.status_code} - {response.text}")

def send_qa_request(question, model=None, stream=False, provider="ollama", temperature=0.7, max_tokens=None):
    """Send a question to the QA endpoint"""
    url = "http://localhost:8000/qa"
    
    # Build request payload
    headers = {"Content-Type": "application/json"}
    data = {
        "question": question,
        "stream": stream,
        "provider": provider
    }
    
    # Only include optional parameters when provided
    if model:
        data["model"] = model
    if temperature != 0.7:
        data["temperature"] = temperature
    if max_tokens:
        data["max_tokens"] = max_tokens
    
    print(f"Sending QA request to {url} with stream={stream}, provider={provider}, model={model or 'default'}")
    
    # Send request and process response
    response = requests.post(url, headers=headers, json=data, stream=stream)
    
    if response.status_code == 200:
        if stream:
            # Handle streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')
                        try:
                            json_data = json.loads(line_text)
                            if "response" in json_data:
                                print(json_data["response"], end="", flush=True)
                            if json_data.get("done", False):
                                print()
                        except json.JSONDecodeError:
                            print(line_text, end="", flush=True)
                    except Exception as e:
                        print(f"Error processing streaming response: {e}")
        else:
            # Handle complete response with structured output
            try:
                response_json = response.json()
                print(f"Question: {response_json.get('question', '')}")
                print(f"Answer: {response_json.get('answer', '')}")
                print(f"Model: {response_json.get('model', '')}")
                print(f"Provider: {response_json.get('provider', '')}")
            except json.JSONDecodeError:
                print(response.text)
    else:
        print(f"Error: {response.status_code} - {response.text}")

def send_code_analysis_request(file_path=None, code=None, analysis_type="general", model=None, stream=False, provider="ollama"):
    """Send a code analysis request to the API"""
    url = "http://localhost:8000/analyze-code"
    
    # Build request payload
    headers = {"Content-Type": "application/json"}
    data = {
        "stream": stream,
        "provider": provider,
        "analysis_type": analysis_type
    }
    
    # Add file path or code content
    if file_path:
        data["file_path"] = file_path
    if code:
        data["code"] = code
    
    # Only include optional parameters when provided
    if model:
        data["model"] = model
    
    print(f"Sending code analysis request to {url} with stream={stream}, provider={provider}, model={model or 'default'}")
    
    # Send request
    response = requests.post(url, headers=headers, json=data, stream=stream)
    
    if response.status_code == 200:
        if stream:
            # Handle streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')
                        print(line_text, end="", flush=True)
                    except Exception as e:
                        print(f"Error processing streaming response: {e}")
        else:
            # Handle complete response with formatted output
            try:
                response_json = response.json()
                if "response" in response_json:
                    print("\n=== CODE ANALYSIS ===")
                    print(response_json["response"])
                    print("\n=== CODE STRUCTURE ===")
                    print(f"File: {response_json.get('file_path', 'N/A')}")
                    
                    if "code_structure" in response_json:
                        structure = response_json["code_structure"]
                        
                        print("\nClasses:")
                        for cls in structure.get("classes", []):
                            print(f"  - {cls['name']} (line {cls['line_number']})")
                            if cls.get("methods"):
                                print(f"    Methods: {', '.join(cls['methods'])}")
                        
                        print("\nFunctions:")
                        for func in structure.get("functions", []):
                            print(f"  - {func['name']} (line {func['line_number']})")
                        
                        print("\nImports:")
                        for imp in structure.get("imports", []):
                            print(f"  - {imp}")
                else:
                    print(response_json)
            except json.JSONDecodeError:
                print(response.text)
    else:
        print(f"Error: {response.status_code} - {response.text}")

def get_models():
    """Retrieve and display available models from all providers"""
    url = "http://localhost:8000/models"
    response = requests.get(url)
    
    if response.status_code == 200:
        models = response.json()
        print("Available models:")
        print("\nOllama models:")
        for model in models.get("ollama", []):
            print(f"  - {model}")
        print("\nOpenAI models:")
        for model in models.get("openai", []):
            print(f"  - {model}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# CLI command handling
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python send_request.py generate <prompt> [model] [stream] [provider]")
        print("  python send_request.py qa <question> [model] [stream] [provider]")
        print("  python send_request.py analyze-code <file_path> [analysis_type] [model] [stream] [provider]")
        print("  python send_request.py models")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == "generate":
        if len(sys.argv) < 3:
            print("Usage: python send_request.py generate <prompt> [model] [stream] [provider]")
            sys.exit(1)
            
        # Parse arguments
        prompt = sys.argv[2]
        model = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].lower() not in ["true", "false"] else None
        stream = len(sys.argv) > 3 and sys.argv[3].lower() == "true" or len(sys.argv) > 4 and sys.argv[4].lower() == "true"
        provider = sys.argv[5] if len(sys.argv) > 5 else "ollama"
        
        send_request(prompt, model, stream, provider)
        
    elif command == "qa":
        if len(sys.argv) < 3:
            print("Usage: python send_request.py qa <question> [model] [stream] [provider]")
            sys.exit(1)
            
        # Parse arguments
        question = sys.argv[2]
        model = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].lower() not in ["true", "false"] else None
        stream = len(sys.argv) > 3 and sys.argv[3].lower() == "true" or len(sys.argv) > 4 and sys.argv[4].lower() == "true"
        provider = sys.argv[5] if len(sys.argv) > 5 else "ollama"
        
        send_qa_request(question, model, stream, provider)
        
    elif command == "analyze-code":
        if len(sys.argv) < 3:
            print("Usage: python send_request.py analyze-code <file_path> [analysis_type] [model] [stream] [provider]")
            sys.exit(1)
            
        # Parse arguments
        file_path = sys.argv[2]
        
        # Check if there are additional parameters
        if len(sys.argv) > 3:
            # Third parameter could be analysis_type, model, or stream
            if sys.argv[3].lower() in ["true", "false"]:
                analysis_type = "general"
                stream = sys.argv[3].lower() == "true"
                model = None
                provider_idx = 4
            else:
                analysis_type = sys.argv[3]
                
                # Check if fourth parameter exists and if it's a model or stream flag
                if len(sys.argv) > 4:
                    if sys.argv[4].lower() in ["true", "false"]:
                        model = None
                        stream = sys.argv[4].lower() == "true"
                        provider_idx = 5
                    else:
                        model = sys.argv[4]
                        # Check if fifth parameter exists and is a stream flag
                        stream = len(sys.argv) > 5 and sys.argv[5].lower() == "true"
                        provider_idx = 6 if stream else 5
                else:
                    model = None
                    stream = False
                    provider_idx = None
        else:
            analysis_type = "general"
            model = None
            stream = False
            provider_idx = None
            
        # Get provider if specified
        provider = sys.argv[provider_idx] if provider_idx and len(sys.argv) > provider_idx else "ollama"
        
        print(f"Analyzing code with parameters: file_path={file_path}, analysis_type={analysis_type}, model={model}, stream={stream}, provider={provider}")
        send_code_analysis_request(file_path=file_path, analysis_type=analysis_type, model=model, stream=stream, provider=provider)
        
    elif command == "models":
        get_models()
        
    else:
        print("Unknown command. Available commands: generate, qa, models")
        sys.exit(1)