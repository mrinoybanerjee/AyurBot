import torch
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

class Evaluate:
    """
    A class to evaluate the performance of the RAG model by comparing the generated answers.
    """
    def __init__(self):
        # Load the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def cosine_similarity(self, embedding1, embedding2):
        """
        Calculate the cosine similarity between two embeddings.

        :param embedding1: The first embedding.
        :param embedding2: The second embedding.
        :return: The cosine similarity between the two embeddings.
        """
        return 1 - cosine(embedding1, embedding2)

    def calculate_similarity_scores(self, true_answer, rag_answer, non_rag_answer):
        """
        Calculate the similarity scores between the true answer and the RAG and non-RAG based answers.

        :param true_answer: The true answer to the question.
        :param rag_answer: The answer generated using the RAG model.
        :param non_rag_answer: The answer generated using a non-RAG model.
        :return: A tuple containing the similarity scores for the RAG and non-RAG answers.
        """
        # Encode the true answer and the generated answers
        true_answer_embedding = self.model.encode(true_answer, convert_to_tensor=False)
        rag_answer_embedding = self.model.encode(rag_answer, convert_to_tensor=False)
        non_rag_answer_embedding = self.model.encode(non_rag_answer, convert_to_tensor=False)

        # Calculate the cosine similarity between the true answer and the generated answers
        rag_similarity_score = self.cosine_similarity(true_answer_embedding, rag_answer_embedding)
        non_rag_similarity_score = self.cosine_similarity(true_answer_embedding, non_rag_answer_embedding)

        return rag_similarity_score, non_rag_similarity_score


