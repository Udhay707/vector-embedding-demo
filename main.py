import google.generativeai as genai
from dotenv import load_dotenv
import os
from semantic_search import search_documents

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
system_prompt = None

def get_system_prompt(prompt_file_path):
    global system_prompt
    if system_prompt != None:
        return system_prompt
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            system_prompt = file.read().strip()
        return system_prompt
    except FileNotFoundError:
        print(f"Warning: Prompt file '{prompt_file_path}' not found. Using default prompt.")
        return "You are a helpful AI assistant. Answer questions based on the provided context."
    except Exception as e:
        print(f"Error loading prompt file: {e}")
        return "You are a helpful AI assistant. Answer questions based on the provided context."


# process_and_store_documents(os.getenv("DOCUMENT_FOLDER_PATH"))
def get_response(user_input):
    context_document = search_documents(user_input)
    context = str(context_document[0]["text"])
    model = genai.GenerativeModel(os.getenv("LLM_MODEL"))
    sys_prompt = get_system_prompt(os.getenv("PROMPT_FOLDER_PATH")+"/user_query_prompt.txt")
    full_prompt = f"""
        {sys_prompt}
        CONTEXT FROM DOCUMENTS:
        {context}
        USER QUESTION: {user_input}
        Please answer the user's question based on the provided context. If the context doesn't contain enough information to answer the question, please say so clearly.
        """
    result = model.generate_content(full_prompt)
    print(result.text)
    


while(True):
    user_input = str(input("Type your question, or exit to quit\n"))
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    get_response(user_input.strip())

