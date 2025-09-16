from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai


load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize embedding model
genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))

index_name = "developer-quickstart-py"

if not pc.has_index(index_name):
    print("Creating new index...")
    pc.create_index(
        name=index_name,
        dimension=768,  # Match your embedding model's dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

def process_and_store_documents(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: Folder path {folder_path} does not exist")
        return
        
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    if not content.strip():
                        print(f"Skipping empty file: {filename}")
                        continue
                        
                    print(f"Processing document: {filename}")

                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=content,
                        task_type="RETRIEVAL_DOCUMENT"
                    )

                    # Generate embeddings
                    embeddings = result["embedding"]
                    
                    # Ensure embeddings are in the right format
                    if isinstance(embeddings, np.ndarray):
                        embeddings = embeddings.tolist()

                    record = {
                        "id": filename.replace('.txt', ''),
                        "values": embeddings,  # Actual embedding values
                        "metadata": {
                            "chunk_text": content,
                            "doc_name": filename
                        }
                    }
                    
                    response = index.upsert(vectors=[record])
                    print(f"Upserted {filename}: {response}")
                    
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except UnicodeDecodeError:
                print(f"Error decoding file {filename}. Try different encoding.")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

def search_documents(query, top_k=2):

    query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
    
    print(query_embedding)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"][0]["metadata"]["chunk_text"]



# Run the processing
# document_folder = os.getenv("DOCUMENT_FOLDER_PATH")
# if document_folder:
#     process_and_store_documents(document_folder)
# else:
#     print("DOCUMENT_FOLDER_PATH environment variable not set")