import streamlit as st
import os
from dotenv import load_dotenv
from model import RAGModel

# Load the environment variables
load_dotenv()

MONGODB_CONNECTION_STRING = str(os.getenv("MONGODB_URI"))
MONGODB_DATABASE = str(os.getenv("MONGODB_DATABASE"))
MONGODB_COLLECTION = str(os.getenv("MONGODB_COLLECTION"))
API_TOKEN = str(os.getenv("API_TOKEN"))

# Initialize the RAGModel with your MongoDB and Replicate settings
rag_model = RAGModel(
    mongo_connection_string=MONGODB_CONNECTION_STRING,
    mongo_database=MONGODB_DATABASE,
    mongo_collection=MONGODB_COLLECTION,
    api_token=API_TOKEN
)

st.set_page_config(page_title="AyurGPT", page_icon="🌿", layout="wide")

# Define your query
query = st.text_input("Message AyurGPT: ")

# Perform semantic search and generate an answer
answer = rag_model.generate_answer(query)

# Print the generated answer

st.write("Generated Answer:", answer)

