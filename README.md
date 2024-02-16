# AyurBot


## Project Overview

"AyurBot: Your Digital Guide to Ancient Ayurvedic Wisdom"

AyurBot is a chatbot designed to offer personalized Ayurvedic health advice. Leveraging the ancient Indian wisdom of Ayurveda, this bot provides users with lifestyle and dietary recommendations based on their unique body constitution (Prakriti). Whether you're looking for natural remedies, daily routines, or nutritional guidance, AyurBot is here to support your journey towards holistic well-being.

## Installation

To get AyurBot up and running on your local machine, follow these steps:

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/AyurBot.git
cd AyurBot
```

2. **Set up a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install required dependencies:**

```bash
pip install -r requirements.txt
```

## Configuration

Before launching AyurBot, configure the environment variables:

1. Rename the `.env.example` file to `.env`.
2. Fill in the necessary values in `.env`, such as API keys, database connection strings, or other configuration details specific to your environment.

## Usage

To start AyurBot, navigate to the `src` directory and run the main script:

```bash
cd src
python main.py
```

Follow the on-screen instructions to interact with the bot.

## Data Management

The `data` directory contains essential datasets and resources used by AyurBot. Here, you'll find information on herbs, dietary recommendations, and Ayurvedic practices. Ensure this data is updated and accurate to enhance the bot's effectiveness.

## Contact

For support, further information, or to contribute to the project, please reach out to me at mrinoybanerjee@gmail.com.

## Project Walkthrough

### AyurBot: Using RAG Based LLM

This section provides a step-by-step guide to integrating a Retrieval-Augmented Generation (RAG) based Large Language Model (LLM) into AyurBot, enhancing its ability to provide accurate and relevant Ayurvedic health advice.


#### Text Extraction from Book

1. **Extract Data**: Functionality is provided to extract text from PDF files, which involves:
   - Opening the PDF file.
   - Removing headers/footers to clean the text.
   - Saving the cleaned text for further processing.

#### Store Chunks in MongoDB Database

1. **Database Connection**: Connect to MongoDB database. Update the connection string as per your setup.
2. **Text Chunking**: Implement a function to chunk text by sentence for efficient storage and retrieval.
3. **Data Storage**: Store these chunks in MongoDB, creating a document for each chunk.


#### Create Word Embeddings

1. **Embeddings Generation**: Using a sentence transformer model, generate embeddings for each text chunk stored in MongoDB.
2. **Database Update**: Update each document in MongoDB with its corresponding embedding for later retrieval.

#### RAG: Semantic Search Retrieval

1. **Semantic Search Functionality**: Implement a function to perform semantic search. This involves generating a query embedding, retrieving and comparing all stored embeddings from MongoDB, and returning the most relevant documents based on similarity.

#### LLM Model

1. **LLM Integration**: Integrate the LLama2 model from Replicate to generate answers based on the context provided by the semantic search.
   - This includes handling context truncation and ensuring the generation of non-empty answers.

## Deployment

The AyurBot application is deployed on Streamlit and is available for public use. This deployment allows users to interact with AyurBot through a user-friendly web interface, making it accessible to anyone interested in Ayurvedic guidance.

## Repository Structure

Below is the structure of the AyurBot repository, providing a clear overview of its organization and contents:

```
.
├── README.md
├── data
│   ├── clean_text
│   │   └── Ayurveda_Book.txt
│   └── pdf
│       └── Ayurveda_Book.pdf
├── notebooks
│   └── rag_model.ipynb
├── requirements.txt
└── src
    ├── __pycache__
    │   └── model.cpython-311.pyc
    ├── app.py
    ├── main.py
    ├── model.py
    └── preprocess.py
```

This structure is designed to facilitate easy navigation and understanding of the project for developers and users alike.
