import google.generativeai as genai
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os


load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

try:
    client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("COLLECTION_NAME")]
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

def search_documents(query, num_results=1):
    """Searches for relevant documents based on a user's query."""    
    try:
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 3, 
                "limit": num_results
            }
        },
        {
            "$project": {
                "text": 1,
                "source_file": 1, 
                "score": { "$meta": "vectorSearchScore" } 
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"Error during vector search: {e}")
        return None

# if __name__ == "__main__":
#     #
#     user_question = "What is aadhar?" 
    
#     print(f"\nSearching for documents related to: '{user_question}'")
    
#     search_results = search_documents(user_question)
    
#     if search_results:
#         print("\n Found relevant document(s):")
#         for doc in search_results:
#             print(f"\n--- Document from: {doc.get('source_file', 'N/A')} ---")
#             print(f"Similarity Score: {doc['score']:.4f}")
#             print(f"Content: {doc['text']}")
#     else:
#         print("\nNo relevant documents found.")