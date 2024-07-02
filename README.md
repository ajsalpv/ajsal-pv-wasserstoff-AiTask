AI Chatbot with Web Scraping and RAG
Welcome to our AI Chatbot project, integrated with web scraping capabilities and powered by a Retrieval-Augmented Generation (RAG) model. This project aims to provide an interactive interface where users can query information related to a website's content, and the chatbot generates relevant responses based on retrieved knowledge.

Features
Web Scraping: Automatically extracts text content from a given URL.
Text Processing: Cleans and preprocesses scraped text for better understanding.
Chunking: Splits large text into smaller chunks suitable for processing by language models.
Embeddings: Converts text chunks into embeddings using Hugging Face models and stores them in a vector store (FAISS).
RAG Model Integration: Uses LangChain's capabilities to set up a Retrieval-Augmented Generation model for answering user queries.
Conversation History: Maintains a memory of user interactions using ConversationBufferMemory from LangChain.
Setup Instructions
To run the project locally, follow these steps:

Clone the Repository:


git clone https://github.com/ajsal/ajsal-pv/wasserstoff/AiTask.git
cd your-repository
Install Dependencies:
Ensure you have Python 3.x installed. Then, install the required Python packages:


pip install -r requirements.txt

Run the Flask Application:
Start the Flask server to run the chatbot application:


python app.py
The application will run on http://localhost:5000.


Usage
Scraping a Website:

Enter a URL in the provided form and click "Scrape and Process URL". The chatbot will scrape the content from the URL.
Asking Questions:

Enter a query in the "Ask a question" form and click "Submit Query". The chatbot will use the RAG model to generate an answer based on the scraped content.
View Conversation History:

The chat interface displays a history of interactions, showing both user queries and bot responses.
Clear Conversation History:

Use the "Clear Conversation History" button to reset the chatbot's memory and start a new session.
Technologies Used

Flask: Web framework for building the server-side application.

LangChain: Framework for integrating language models, handling text processing, embeddings, and conversation memory.

Hugging Face Transformers: Used for embeddings and language model integration.

Beautiful Soup: Python library for web scraping.

FAISS: Library for efficient similarity search and clustering of dense vectors.
