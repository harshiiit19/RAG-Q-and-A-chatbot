# RAG-Q-A-chatbot

This Streamlit application acts as a Loan Approval Q&A Chatbot that allows users to query information about loan applications. It leverages a combination of a local dataset, sentence embeddings, FAISS for efficient similarity search, and the Google Gemini 2.0 Flash API for generating intelligent responses.

The chatbot works by:

Processing Loan Data: It reads a Training Dataset.csv file containing loan application details.

Creating Embeddings and Index: It transforms each loan application into a text document and then generates numerical embeddings for these documents using a SentenceTransformer model (all-MiniLM-L6-v2). These embeddings are stored in a FAISS (Facebook AI Similarity Search) index for fast retrieval.

Contextual Retrieval: When a user asks a question, the application converts the query into an embedding and uses FAISS to find the most relevant loan application documents from its indexed data.

AI-Powered Answering: The retrieved relevant information (context) along with the user's question is then sent to the Gemini 2.0 Flash API. The AI model uses this context to generate a precise answer, explicitly stating if it cannot provide a precise answer based on the available data.

This setup ensures that the chatbot's responses are grounded in the provided loan data, making it a reliable tool for quickly extracting insights from the dataset.
