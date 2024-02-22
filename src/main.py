import sys
sys.path.append('/Users/mrinoyb2/git/AyurBot/src/model.py')
from model import RAGModel
from evaluate import Evaluate
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")
API_TOKEN = os.getenv("API_TOKEN")


def chat_mode():
    """
    Funtion to activate chat mode
    
    :return: None
    """
    # Initialize the RAGModel with your MongoDB and Replicate settings
    rag_model = RAGModel(
        mongo_connection_string=MONGODB_CONNECTION_STRING,
        mongo_database=MONGODB_DATABASE,
        mongo_collection=MONGODB_COLLECTION,
        api_token=API_TOKEN
    )

    # Chat mode
    # Define your query
    query = input("Enter query: ")

    # Perform semantic search and generate an answer
    answer = rag_model.generate_answer(query)

    # Print the generated answer
    print("Generated Answer:", answer)


def eval_mode():
    """
    Function to activate evaluate mode
    
    :return: None
    """
    # Initialize the RAGModel with your MongoDB and Replicate settings
    rag_model = RAGModel(
        mongo_connection_string=MONGODB_CONNECTION_STRING,
        mongo_database=MONGODB_DATABASE,
        mongo_collection=MONGODB_COLLECTION,
        api_token=API_TOKEN
    )

    # Initialize the Evaluate class
    eval = Evaluate()

    # Evaluate mode
    # Define your query
    print("You are now in evaluate mode!")

    query = input("Enter query: ")

    # Enter true answer from the text
    true_answer = input("Enter true answer: ")

    # RAG based answer
    rag_answer = rag_model.generate_RAG_answer(query)

    # Non-RAG based answer
    non_rag_answer = rag_model.generate_non_RAG_answer(query)

    # Similarity scores
    similarity_scores = eval.calculate_similarity_scores(true_answer, rag_answer, non_rag_answer)

    # Print the generated answer
    print("RAG Answer:", rag_answer)
    print()
    print("Non-RAG Answer:", non_rag_answer)
    print()
    print("Similarity Scores:")
    print()
    print("RAG Answer:", similarity_scores[0])
    print("Non-RAG Answer:", similarity_scores[1])


def main():
    # Welcome message
    print("Welcome to AyurGPT! Sharing ancient ayurvedic wisdom with you. Type 1 if you want to go to chat mode or 2 if you want to activate evaluate mode.")

    if input() == "1":
        chat_mode()
    
    else:
        eval_mode()
        

if __name__ == "__main__":
    main()