import sys
sys.path.append('/Users/mrinoyb2/git/AyurBot/src/model.py')
from model import RAGModel
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")
API_TOKEN = os.getenv("API_TOKEN")

def main():
    # Initialize the RAGModel with your MongoDB and Replicate settings
    rag_model = RAGModel(
        mongo_connection_string=MONGODB_CONNECTION_STRING,
        mongo_database=MONGODB_DATABASE,
        mongo_collection=MONGODB_COLLECTION,
        api_token=API_TOKEN
    )

    # Define your query
    query = input("Message AyurBot: ")
    
    # Perform semantic search and generate an answer
    answer = rag_model.generate_answer(query)
    
    # Print the generated answer
    print("Generated Answer:", answer)

if __name__ == "__main__":
    main()
