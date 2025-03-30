from send_request import send_request, get_models

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

if __name__ == "__main__":
    print("Demo Script Execution\n")
    
    print("Basic LLM Demos:")
    demo_streaming_response()
    demo_complete_response()
    
    print("\nOpenAI Integration Demo:")
    print("NOTE: OpenAI demos require a valid API key in config.py or .env file")
    demo_openai_response()
    
    print("\nModel Information:")
    demo_available_models()
