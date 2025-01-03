import streamlit as st
import requests
import pdfplumber
from io import BytesIO
from langchain_community.document_loaders import UnstructuredURLLoader
import pickle
import openai
import faiss
import numpy as np
import configparser

# Load default configuration for API Key
config = configparser.ConfigParser()
config.read('config.ini')
default_api_key = config['openai']['api_key']  # Default API key from config file

class CustomURLLoader:
    # Handles loading and extracting text from PDF files
    def load_pdf(self, url):
        # Request the content of the URL
        response = requests.get(url)
        # Open the PDF and extract text from each page
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text

    # Determines the type of URL (PDF or other) and uses appropriate method to load content
    def load(self, url):
        if url.lower().endswith('.pdf'):
            return self.load_pdf(url)
        else:
            loader = UnstructuredURLLoader(urls=[url])
            return loader.load()[0]

# Setup for the FAISS index used for efficient similarity search of embeddings
dimension = 768
index = faiss.IndexFlatL2(dimension)
try:
    with open("faiss_store_openai.pkl", "rb") as f:
        index = pickle.load(f)
except FileNotFoundError:
    # Initialize a new FAISS index if not found
    pass

# Function to save the FAISS index to disk
def save_faiss_index():
    with open("faiss_store_openai.pkl", "wb") as f:
        pickle.dump(index, f)

# Function to generate embeddings using OpenAI's API
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return np.array(response['data'][0]['embedding'])

# Process a single URL: load content, generate embedding, and add to FAISS index
def process_url(url):
    loader = CustomURLLoader()
    content = loader.load(url)
    if content:
        st.write("Content Loaded Successfully for URL:", url)
        st.text_area("Content Preview", content, height=300)
        embedding = get_embedding(content)
        index.add(embedding.reshape(1, -1))
        save_faiss_index()
        st.success("Content indexed.")
    else:
        st.error("Failed to load content from URL.")

# Main function to run the Streamlit application
def main():
    st.title("Scheme Research Tool")

    # Sidebar input for OpenAI API key with password masking, user input is optional
    user_api_key = st.sidebar.text_input("Enter your OpenAI API key (optional)", type="password")

    # Set the API key for OpenAI requests (use user input if provided, otherwise use default)
    openai.api_key = user_api_key if user_api_key else default_api_key

    # Inputs for processing URLs individually or from a file
    url_input = st.sidebar.text_input("Enter URL")
    uploaded_file = st.sidebar.file_uploader("Upload a file with URLs", type=['txt'])
    process_btn = st.sidebar.button("Process URL or File")

    # Button to process the input URL or URLs from the uploaded file
    if process_btn:
        if url_input:
            process_url(url_input)
        elif uploaded_file:
            for url in uploaded_file.getvalue().decode("utf-8").splitlines():
                process_url(url.strip())

    # Input and process for user queries to find related content based on embeddings
    query = st.text_input("Ask a question")
    if query:
        query_embedding = get_embedding(query)
        D, I = index.search(np.array([query_embedding]), 1)
        st.write(f"Response index: {I[0][0]}, Distance: {D[0][0]}")

if __name__ == "__main__":
    main()
