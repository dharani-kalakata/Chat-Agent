from send_request import send_request, get_models, load_codebase, ask_codebase, generate_tests

def demo_streaming_response():
    print("\n--- Streaming Response (Ollama) ---")
    send_request(prompt="Write a haiku.", model="llama3.1", stream=True, provider="ollama")
    print("\n--- End of Streaming Response ---")

def demo_complete_response():
    print("\n--- Complete Response (Ollama) ---")
    send_request(prompt="Write a haiku.", model="llama3.1", provider="ollama")
    print("\n--- End of Complete Response ---")

def demo_openai_response():
    print("\n--- OpenAI Integration ---")
    send_request(prompt="Write a haiku.", model="gpt-3.5-turbo", provider="openai")
    print("\n--- End of OpenAI Integration ---")

def demo_available_models():
    print("\n--- Available Models ---")
    get_models()
    print("\n--- End of Available Models ---")

def demo_codebase_loading():
    print("\n--- Codebase Loading ---")
    # Load the current project's code
    load_codebase("./")
    print("\n--- End of Codebase Loading ---")

def demo_codebase_qa():
    print("\n--- Codebase Question Answering ---")
    ask_codebase("What does the LangChainService class do in this codebase?", provider="ollama")
    print("\n--- End of Codebase Question Answering ---")

def demo_test_generation():
    print("\n--- Test Generation ---")
    generate_tests("services/langchain_service.py", provider="ollama")
    print("\n--- End of Test Generation ---")

if __name__ == "__main__":
    print("Demo Script Execution\n")
    
    print("Basic LLM Demos:")
    demo_streaming_response()
    demo_complete_response()
    
    print("\nOpenAI Integration Demo:")
    print("NOTE: OpenAI demos require a valid API key in config.py or .env file")
    demo_openai_response()
    
    print("\nCode Analysis Demos:")
    demo_codebase_loading()
    demo_codebase_qa()
    demo_test_generation()
    
    print("\nModel Information:")
    demo_available_models()
