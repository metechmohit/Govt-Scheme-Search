import streamlit as st
import requests
import pdfplumber
from io import BytesIO
from langchain_community.document_loaders import UnstructuredURLLoader
import pickle
import openai
import faiss
import numpy as np

# Class for loading URLs, handling both PDFs and other URL types.
class CustomURLLoader:
    # Method to load and extract text from PDF URLs.
    def load_pdf(self, url):
        response = requests.get(url)
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text

    # Method to load content from URLs. Checks if URL is a PDF or standard webpage.
    def load(self, url):
        if url.lower().endswith('.pdf'):
            return self.load_pdf(url)
        else:
            loader = UnstructuredURLLoader(urls=[url])
            return loader.load()[0]

# FAISS index for efficient similarity search of embeddings.
dimension = 768
index = faiss.IndexFlatL2(dimension)
try:
    with open("faiss_store_openai.pkl", "rb") as f:
        index = pickle.load(f)
except FileNotFoundError:
    # Handle missing FAISS index file by initializing a new index.
    pass

# Function to save the FAISS index to disk.
def save_faiss_index():
    with open("faiss_store_openai.pkl", "wb") as f:
        pickle.dump(index, f)

# Function to generate embeddings using OpenAI's API.
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return np.array(response['data'][0]['embedding'])

# Function to process a single URL, load content, and generate embeddings.
def process_url(url):
    loader = CustomURLLoader()
    content = loader.load(url)
    if content:
        st.write("Content Loaded Successfully for URL:", url)
        st.text_area("Content Preview", content[:500], height=300)
        embedding = get_embedding(content)
        index.add(embedding.reshape(1, -1))
        save_faiss_index()
        st.success("Content indexed.")
    else:
        st.error("Failed to load content from URL.")

# Main function to run the Streamlit application.
def main():
    st.title("Scheme Research Tool")
    
    # Sidebar input for OpenAI API key with password masking.
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        openai.api_key = api_key  # Set the API key for OpenAI requests.

    # Sidebar inputs for URL and file uploading.
    url_input = st.sidebar.text_input("Enter URL")
    uploaded_file = st.sidebar.file_uploader("Upload a file with URLs", type=['txt'])
    process_btn = st.sidebar.button("Process URL or File")

    # Process the provided URL or file.
    if process_btn:
        if url_input:
            process_url(url_input)
        elif uploaded_file:
            for url in uploaded_file.getvalue().decode("utf-8").splitlines():
                process_url(url.strip())

    # Input field for asking questions and retrieving information based on embeddings.
    query = st.text_input("Ask a question")
    if query:
        query_embedding = get_embedding(query)
        D, I = index.search(np.array([query_embedding]), 1)
        st.write(f"Response index: {I[0][0]}, Distance: {D[0][0]}")

if __name__ == "__main__":
    main()
