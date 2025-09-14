from dotenv import load_dotenv
import google.generativeai as genai
from pymongo.mongo_client import MongoClient
import os

load_dotenv()

try:
    client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
    #
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("COLLECTION_NAME")]
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()


def get_embedding(text):
    """Generates an embedding for a given text using Google's model."""
    try:
        genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding for text: '{text[:20]}...'. Error: {e}")
        return None
    
def process_and_store_documents(folder_path):
    """Reads all .txt files in a folder, generates embeddings, and stores them in MongoDB."""
    documents_to_insert = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f" Processing document: {filename}")
                    embedding = get_embedding(content)
                    
                    if embedding:
                        document = {
                            "source_file": filename,
                            "text": content,
                            "embedding": embedding
                        }
                        documents_to_insert.append(document)
            except Exception as e:
                print(f"Error reading or processing file {filename}. Error: {e}")

    if documents_to_insert:
        print(f"\n✅ Found {len(documents_to_insert)} documents. Inserting into MongoDB...")
        collection.insert_many(documents_to_insert)
        print("🎉 All documents have been successfully embedded and stored in the database!")
    else:
        print("🤷 No documents were processed or found.")


if __name__ == "__main__":
    process_and_store_documents(os.getenv("DOCUMENT_FOLDER_PATH"))