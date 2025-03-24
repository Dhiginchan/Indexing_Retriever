from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Get Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 1️⃣ **Create Pinecone instance**
pc = Pinecone(api_key=PINECONE_API_KEY)


# 3️⃣ **Connect to the Existing Index**
index = pc.Index(PINECONE_INDEX_NAME)

# 4️⃣ **Load the Same Embeddings Used in Indexing**
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def retrieve_answer(query, top_k=3):
    """
    This function takes a user question, converts it into a vector,
    searches in Pinecone, and returns the best-matching text chunks.
    """
    print(f"\n🔍 Searching for: {query}")

    # 5️⃣ **Convert the Query to a Vector**
    query_vector = embeddings.embed_query(query)

    # 6️⃣ **Search in Pinecone**
    search_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # 7️⃣ **Display the Top Results**
    if "matches" in search_results:
        print("\n🎯 Top Matching Results:\n")
        for i, match in enumerate(search_results["matches"]):
            print(f"📌 Result {i+1} (Score: {match['score']}):\n{match['metadata']['text']}\n")
    else:
        print("❌ No matches found!")

# Example Query
retrieve_answer("which time period it has built?")
