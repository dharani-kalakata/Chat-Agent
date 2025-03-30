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

def load_codebase(path, file_extension="*.py"):
    """Load and index a codebase for analysis"""
    url = "http://localhost:8000/codebase/load"
    
    # Build request payload
    headers = {"Content-Type": "application/json"}
    data = {
        "path": path,
        "file_extension": file_extension
    }
    
    print(f"Loading codebase from {path}...")
    
    # Send request
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result.get('message', '')}")
        print(f"Loaded {result.get('file_count', 0)} files with {result.get('chunk_count', 0)} chunks")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def ask_codebase(question, model=None, stream=False, provider="ollama", temperature=0.7, max_tokens=None):
    """Ask a question about the loaded codebase"""
    url = "http://localhost:8000/codebase/question"
    
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
    
    print(f"Asking codebase: {question}")
    
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
            # Handle complete response
            try:
                result = response.json()
                print("\nQuestion:")
                print(result.get("question", ""))
                print("\nAnswer:")
                print(result.get("answer", ""))
                print("\nSources:")
                for source in result.get("sources", []):
                    print(f"- {source}")
            except json.JSONDecodeError:
                print(response.text)
    else:
        print(f"Error: {response.status_code} - {response.text}")

def generate_tests(file_path, model=None, stream=False, provider="ollama", temperature=0.7, max_tokens=None):
    """Generate pytest test cases for a Python file"""
    url = "http://localhost:8000/codebase/generate-tests"
    
    # Build request payload
    headers = {"Content-Type": "application/json"}
    data = {
        "file_path": file_path,
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
    
    print(f"Generating tests for: {file_path}")
    
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
            # Handle complete response
            try:
                result = response.json()
                print("\nTest code for:", result.get("file_path", ""))
                print(f"Suggested filename: {result.get('suggested_test_filename', '')}")
                print("\n" + result.get("test_code", ""))
            except json.JSONDecodeError:
                print(response.text)
    else:
        print(f"Error: {response.status_code} - {response.text}")

# CLI command handling
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python send_request.py generate <prompt> [model] [stream] [provider]")
        print("  python send_request.py models")
        print("  python send_request.py load-codebase <path> [file_extension]")
        print("  python send_request.py ask-codebase <question> [model] [stream] [provider]")
        print("  python send_request.py generate-tests <file_path> [model] [stream] [provider]")
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
        
    elif command == "models":
        get_models()
        
    elif command == "load-codebase":
        if len(sys.argv) < 3:
            print("Usage: python send_request.py load-codebase <path> [file_extension]")
            sys.exit(1)
            
        path = sys.argv[2]
        file_extension = sys.argv[3] if len(sys.argv) > 3 else "*.py"
        
        load_codebase(path, file_extension)
        
    elif command == "ask-codebase":
        if len(sys.argv) < 3:
            print("Usage: python send_request.py ask-codebase <question> [model] [stream] [provider]")
            sys.exit(1)
            
        question = sys.argv[2]
        model = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].lower() not in ["true", "false"] else None
        stream = len(sys.argv) > 3 and sys.argv[3].lower() == "true" or len(sys.argv) > 4 and sys.argv[4].lower() == "true"
        provider = sys.argv[5] if len(sys.argv) > 5 else "ollama"
        
        ask_codebase(question, model, stream, provider)
        
    elif command == "generate-tests":
        if len(sys.argv) < 3:
            print("Usage: python send_request.py generate-tests <file_path> [model] [stream] [provider]")
            sys.exit(1)
            
        file_path = sys.argv[2]
        model = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].lower() not in ["true", "false"] else None
        stream = len(sys.argv) > 3 and sys.argv[3].lower() == "true" or len(sys.argv) > 4 and sys.argv[4].lower() == "true"
        provider = sys.argv[5] if len(sys.argv) > 5 else "ollama"
        
        generate_tests(file_path, model, stream, provider)
        
    else:
        print("Unknown command. Available commands: generate, models, load-codebase, ask-codebase, generate-tests")
        sys.exit(1)