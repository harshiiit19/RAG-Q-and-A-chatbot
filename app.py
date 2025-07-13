import streamlit as st
import pandas as pd
import faiss
import numpy as np
import pickle
import requests
import os
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DATASET_PATH = 'Training Dataset.csv'
FAISS_INDEX_PATH = 'loan_faiss_index.bin'
DOCUMENTS_PATH = 'loan_documents.pkl'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = "AIzaSyAnF0YIQXSZhHzktuBAEbnCiDZZgQgeZ-Y" 

# --- Helper Functions ---

@st.cache_resource 
def load_embedding_model():
    """Loads the pre-trained sentence transformer model."""
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

def create_documents_and_index(df, model):
    """
    Processes the DataFrame, creates text documents, generates embeddings,
    builds a FAISS index, and saves them.
    """

    documents = []
    # Create a text document for each row, describing the loan application
    for index, row in df.iterrows():
        # Handle potential NaN values gracefully for display
        gender = row['Gender'] if pd.notna(row['Gender']) else 'N/A'
        married = row['Married'] if pd.notna(row['Married']) else 'N/A'
        dependents = row['Dependents'] if pd.notna(row['Dependents']) else 'N/A'
        education = row['Education'] if pd.notna(row['Education']) else 'N/A'
        self_employed = row['Self_Employed'] if pd.notna(row['Self_Employed']) else 'N/A'
        applicant_income = row['ApplicantIncome'] if pd.notna(row['ApplicantIncome']) else 'N/A'
        coapplicant_income = row['CoapplicantIncome'] if pd.notna(row['CoapplicantIncome']) else 'N/A'
        loan_amount = row['LoanAmount'] if pd.notna(row['LoanAmount']) else 'N/A'
        loan_amount_term = row['Loan_Amount_Term'] if pd.notna(row['Loan_Amount_Term']) else 'N/A'
        credit_history = row['Credit_History'] if pd.notna(row['Credit_History']) else 'N/A'
        property_area = row['Property_Area'] if pd.notna(row['Property_Area']) else 'N/A'
        loan_status = row['Loan_Status'] if pd.notna(row['Loan_Status']) else 'N/A'

        doc_text = (
            f"Loan Application ID: {row['Loan_ID']}. "
            f"Applicant details: Gender: {gender}, Married: {married}, Dependents: {dependents}, "
            f"Education: {education}, Self-Employed: {self_employed}. "
            f"Financials: Applicant Income: {applicant_income}, Coapplicant Income: {coapplicant_income}, "
            f"Loan Amount: {loan_amount}, Loan Term: {loan_amount_term} months. "
            f"Other info: Credit History: {credit_history}, Property Area: {property_area}. "
            f"Loan Status: {loan_status}."
        )
        documents.append(doc_text)

    # Generate embeddings for all documents
    document_embeddings = model.encode(documents, show_progress_bar=True)
    document_embeddings = np.array(document_embeddings).astype('float32')

    # Build FAISS index
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) 
    index.add(document_embeddings)

    # Save documents and index
    with open(DOCUMENTS_PATH, 'wb') as f:
        pickle.dump(documents, f)
    faiss.write_index(index, FAISS_INDEX_PATH)

    return documents, index

def load_documents_and_index():
    """Loads pre-saved documents and FAISS index."""
    with open(DOCUMENTS_PATH, 'rb') as f:
        documents = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_PATH)
    return documents, index

def retrieve_context(query, index, documents, model, k=5): 
    """Retrieves top-k most relevant documents based on query."""
    query_embedding = model.encode([query]).astype('float32')
    D, I = index.search(query_embedding, k) 
    
    retrieved_docs = [documents[i] for i in I[0]]
    return "\n".join(retrieved_docs)

def generate_response_with_gemini(user_query, context):
    """Generates a response using the Gemini 2.0 Flash API."""
    
    
    # Check if API key is available
    if not GEMINI_API_KEY:
        st.error("Gemini API Key is missing. Please ensure it's provided by the Canvas environment or set it.")
        return "Sorry, I cannot connect to the AI model without an API key. Please check your environment setup."

    prompt = (
        f"You are a helpful assistant providing information about loan applications. "
        f"Strictly use ONLY the following provided loan application data to answer the user's question. "
        f"If the provided data does not contain enough information to answer precisely, "
        f"state clearly that you cannot provide a precise answer based on the available data. "
        f"Do not make up information.\n\n"
        f"Loan Application Data:\n{context}\n\n"
        f"User Question: {user_query}\n\n"
        f"Answer:"
    )

    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }
    
    # Append API key to URL if available, otherwise it will be handled by Canvas runtime
    api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    try:
        response = requests.post(api_url_with_key, headers=headers, json=payload)
        response.raise_for_status() 
        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and \
           result['candidates'][0]['content'].get('parts') and \
           result['candidates'][0]['content']['parts'][0].get('text'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            st.error("Gemini API response did not contain expected text.")
            st.json(result) # Display the full response for debugging
            return "Sorry, I couldn't generate a response from the AI model. The response structure was unexpected."
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Gemini API: {e}. This often indicates an issue with your API key or network.")
        return "Sorry, there was an issue connecting to the AI model. Please check your API key and network connection."
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred while processing the AI response."


# --- Streamlit App Layout ---
st.set_page_config(page_title="Loan Approval Q&A Chatbot", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .response-box {
        background-color: #d4edda; /* Light green background */
        color: #155724; /* Dark green text */
        border-left: 5px solid #28a745; /* Green border */
        padding: 15px;
        margin-top: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Loan Approval Q&A Chatbot")
st.markdown(
    "Ask me questions about the loan application data from the Dataset. "
    "And I will retrieve relevant information "
    "and use AI to generate an answer."
)

# --- Load/Create Data and Index ---
embedding_model = load_embedding_model()

documents = None
faiss_index = None

# Check if FAISS index and documents are already saved
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH):
    documents, faiss_index = load_documents_and_index()
else:
    # Load the dataset
    try:
        df = pd.read_csv(DATASET_PATH)
        documents, faiss_index = create_documents_and_index(df, embedding_model)
    except FileNotFoundError:
        st.error(
            f"Error: '{DATASET_PATH}' not found. "
            "Please ensure the dataset CSV is in the same directory as app.py."
        )
        st.stop() # Stop the app if the file is not found
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        st.stop()

# --- Chat Interface ---
user_query = st.text_input("Enter your question:", placeholder="e.g., What is the average loan amount for graduates?")

if st.button("Get Answer"):
    if user_query and documents and faiss_index:
        with st.spinner("Processing your request..."):
            # Step 1: Retrieve context
            retrieved_context = retrieve_context(user_query, faiss_index, documents, embedding_model)
            
            # Step 2: Generate response using LLM
            ai_response = generate_response_with_gemini(user_query, retrieved_context)
            
            st.markdown("### Chatbot Response:")
            st.markdown(f"<div class='response-box'>{ai_response}</div>", unsafe_allow_html=True)
            
    elif not user_query:
        st.warning("Please enter a question.")
    else:
        st.error("Chatbot not fully initialized. Please check logs for errors.")

