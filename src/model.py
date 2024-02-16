from sentence_transformers import SentenceTransformer
import pymongo
import numpy as np
from scipy.spatial.distance import cosine
import replicate
from replicate.client import Client

class RAGModel:
    """
    A class to perform Retrieval-Augmented Generation (RAG) for generating answers
    based on semantic search within a MongoDB database and using the LLaMA2 model
    from Replicate for text generation.
    """
    def __init__(self, mongo_connection_string, mongo_database, mongo_collection, api_token):
        """
        Initializes the RAGModel with MongoDB connection settings and the API token for Replicate.
        
        :param mongo_connection_string: MongoDB connection string.
        :param mongo_database: MongoDB database name.
        :param mongo_collection: MongoDB collection name.
        :param api_token: Replicate API token.
        """
        self.mongo_connection_string = mongo_connection_string
        self.mongo_database = mongo_database
        self.mongo_collection = mongo_collection
        self.api_token = api_token
        self.client = pymongo.MongoClient(self.mongo_connection_string)
        self.db = self.client[self.mongo_database]
        self.chunks_collection = self.db[self.mongo_collection]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load sentence transformer model for embeddings

    def semantic_search(self, query, top_k=5):
        """
        Performs semantic search to find the most relevant text chunks based on the query.
        
        :param query: The query string for which to find relevant documents.
        :param top_k: The number of top results to return.
        :return: A list of tuples containing the document ID, similarity score, and text for each result.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        similarities = []
        for document in self.chunks_collection.find():
            doc_embedding = np.array(document['embedding'])
            similarity = 1 - cosine(query_embedding, doc_embedding)
            similarities.append((document['_id'], similarity, document['text']))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def generate_answer(self, question, max_context_length=1000):
        """
        Generates an answer to a given question using the best context found by semantic search
        and the LLaMA2 model from Replicate.
        
        :param question: The question to generate an answer for.
        :param max_context_length: Maximum length of the context to be considered.
        :return: Generated answer as a string.
        """
        context_results = self.semantic_search(question, top_k=1)
        if context_results:
            context = context_results[0][2]
            if len(context) > max_context_length:
                context = context[:max_context_length]
            prompt = f"[INST]\nQuestion: {question}\nContext: {context}\n[/INST]"
        else:
            prompt = f"[INST]\nQuestion: {question}\n[/INST]"

        replicate_client = Client(api_token=self.api_token)
        output = replicate_client.run(
            "nwhitehead/llama2-7b-chat-gptq:8c1f632f7a9df740bfbe8f6b35e491ddfe5c43a79b43f062f719ccbe03772b52",
            input={
                "seed": -1,
                "top_k": 20,
                "top_p": 1,
                "prompt": prompt,
                "max_tokens": 1024,
                "min_tokens": 1,
                "temperature": 0.5,
                "repetition_penalty": 1
            }
        )

        answer = ""
        # Concatenate the output items into a single string
        for item in output:
            answer += item

        # Handle the case where the answer is empty
        if not answer:
            answer = "Sorry, I don't have an answer for that."

        return answer
