import fitz  # PyMuPDF
import google.generativeai as genai

# Initialize Gemini API
def initialize_gemini(api_key):
    genai.configure(api_key=api_key)

def extract_text_from_pdf(pdf_path):
    """ Extract all text from a PDF file. """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def ask_gemini(question, context):
    """ Ask Gemini a question with PDF context. """
    model = genai.GenerativeModel('gemini-1.5-pro')  # Use the correct model name
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

def chatbot(pdf_path, question, api_key):
    initialize_gemini(api_key)
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Optional: Limit context size if PDF is very large
    max_chars = 3000
    if len(pdf_text) > max_chars:
        pdf_text = pdf_text[:max_chars] + "..."
    
    answer = ask_gemini(question, pdf_text)
    return answer

# Example usage
if __name__ == "__main__":
    api_key = input("Enter your Gemini API Key: ")
    pdf_path = input("Enter the PDF file path: ")
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        result = chatbot(pdf_path, question, api_key)
        print("Answer:", result)

