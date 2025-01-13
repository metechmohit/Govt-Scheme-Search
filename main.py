import streamlit as st
import requests
# from io import BytesIO
from pydantic import BaseModel
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
import openai
import faiss
import numpy as np
import configparser

# Load default configuration for API Key
config = configparser.ConfigParser()
config.read('config.ini')
# default_api_key = config['openai']['api_key']

## Take default api_key saved in st.secrets
default_api_key = st.secrets["default_api_key"]   


# Initialize session state
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embeddings_dict' not in st.session_state:
    st.session_state.embeddings_dict = {}
if 'faiss_index' not in st.session_state:
    dimension = 1536
    st.session_state.faiss_index = faiss.IndexFlatL2(dimension)

class CustomURLLoader:
    def load_pdf(self, url):
        try:
            loader = PyPDFLoader(url)
            pages = loader.load()
            
            # Combine all page contents with proper formatting
            text_chunks = []
            for page in pages:
                # Clean and format the page content
                page_text = page.page_content.strip()
                if page_text:  # Only add non-empty pages
                    text_chunks.append(page_text)
            
            # Join with double newlines and ensure proper spacing
            text = "\n\n".join(text_chunks)
            return text
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return ""

    def load(self, url):
        if url.lower().endswith('.pdf'):
            return self.load_pdf(url)
        else:
            try:
                loader = UnstructuredURLLoader(urls=[url])
                docs = loader.load()
                return docs[0].page_content if docs else ""
            except Exception as e:
                st.error(f"Error loading URL: {str(e)}")
                return ""

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Explicit separators for better chunking
    )
    chunks = text_splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]  # Remove empty chunks

def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return np.array(response['data'][0]['embedding'])
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def process_url(url):
    try:
        # Clear previous data when processing new URL
        st.session_state.chunks = []
        st.session_state.faiss_index = faiss.IndexFlatL2(1536)
        
        # Load and process the content
        loader = CustomURLLoader()
        content = loader.load(url)
        
        if content:
            st.write("Content Loaded Successfully for URL:", url)
            # Show preview of first 500 characters
            st.text_area("Content Preview", content[:500] + "...", height=150)
            
            # Generate chunks
            new_chunks = chunk_text(content)
            
            if not new_chunks:
                st.error("No valid content chunks were generated")
                return
                
            # Store chunks and process embeddings
            for chunk in new_chunks:
                embedding = get_embedding(chunk)
                if embedding is not None:
                    st.session_state.chunks.append(chunk)
                    st.session_state.faiss_index.add(embedding.reshape(1, -1))
            
            st.success(f"Successfully processed {len(st.session_state.chunks)} chunks from the document")
            st.info(f"Total chunks stored: {len(st.session_state.chunks)}")
        else:
            st.error("Failed to load content from URL")
            
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")

def main():
    st.title("Scheme Research Tool")

    # Sidebar setup
    user_api_key = st.sidebar.text_input("Enter your OpenAI API key (optional)", type="password")
    openai.api_key = user_api_key if user_api_key else default_api_key

    # URL processing section
    url_input = st.sidebar.text_input("Enter URL")
    uploaded_file = st.sidebar.file_uploader("Upload a file with URLs", type=['txt'])
    process_btn = st.sidebar.button("Process URL or File")

    # Process URLs
    if process_btn:
        if url_input:
            process_url(url_input)
        elif uploaded_file:
            for url in uploaded_file.getvalue().decode("utf-8").splitlines():
                process_url(url.strip())

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Query handling
    query = st.chat_input("Ask a question about the scheme")
    
    if query:
        with st.chat_message("user"):
            st.write(query)
        
        st.session_state.chat_history.append({"role": "user", "content": query})

        if len(st.session_state.chunks) == 0:
            with st.chat_message("assistant"):
                response = "Please process a document first before asking questions."
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            # Generate query embedding
            query_embedding = get_embedding(query)
            if query_embedding is not None:
                # Perform semantic search
                k = min(5, len(st.session_state.chunks))
                distances, indices = st.session_state.faiss_index.search(np.array([query_embedding]), k)
                
                # Get relevant chunks
                relevant_chunks = [st.session_state.chunks[i] for i in indices[0] if i < len(st.session_state.chunks)]
                context = "\n\n".join(relevant_chunks)

                # Debug info
                with st.expander("Debug Info"):
                    st.write("Number of chunks found:", len(relevant_chunks))
                    st.write("Context length:", len(context))
                    st.write("Closest chunk preview:", relevant_chunks[0][:200] if relevant_chunks else "No chunks found")

                messages = [
                    {"role": "system", "content": (
                        "You are a helpful assistant that provides information about government schemes. "
                        "Base your answers specific to the context provided and try to answer in points. If the information isn't in the "
                        "context, say so clearly. Be concise and informative."
                    )},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ]

                with st.spinner("Generating response..."):
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=messages,
                            max_tokens=500,
                            temperature=0.7
                        )
                        
                        answer = response.choices[0].message.content
                        
                        with st.chat_message("assistant"):
                            st.write(answer)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()