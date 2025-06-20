import google.generativeai as genai

def list_available_models(api_key):
    # Initialize Gemini API
    genai.configure(api_key=api_key)

    # List all models
    models = genai.list_models()

    # Show available models
    print("Available models:")
    for model in models:
        print(model.name)

# Replace with your actual API key
api_key = ""
list_available_models(api_key)

