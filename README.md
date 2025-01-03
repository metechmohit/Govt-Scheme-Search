## Scheme Research Tool

### Overview
The Scheme Research Tool is designed to facilitate the analysis of government scheme articles. By inputting a URL or uploading text files containing URLs, users can access automated summaries and interactive query responses about various schemes. The application highlights Scheme Benefits, Application Processes, Eligibility Criteria, and Required Documents.

### Features
- **URL Input**: Users can input URLs directly into the sidebar to fetch content from specific scheme articles.
- **File Upload**: Users can upload text files containing multiple URLs for batch processing.
- **Content Summarization**: Automated generation of concise summaries covering key scheme details.
- **Interactive Query System**: Engage with the application through queries and receive summarized responses based on the embedded content.
- **Efficient Data Handling**: Utilizes FAISS for swift and effective retrieval of relevant information through embedding vectors.
- **Local Storage of Embeddings**: Embeddings are stored in a local FAISS index for quick access and efficient querying.

### Technologies Used
- **Streamlit**: For creating the web application interface.
- **LangChain's UnstructuredURL Loader**: For processing article content.
- **OpenAI Embeddings**: To generate text embeddings.
- **FAISS (Facebook AI Similarity Search)**: For managing and querying embedding vectors.

### Installation and Setup
1. **Clone the repository**:
   ```bash
   git clone https://your-repository-url.com
   cd your-repository-directory
   ```

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration**:
   - Place your OpenAI API key in the `.config` file under the `[openai]` section:
     ```
     [openai]
     api_key = your_api_key
     ```

4. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

### Usage
- **Launch the application**: Open your web browser and visit `http://localhost:8501` or the URL provided by Streamlit after running the command.
- **Input a URL or Upload a File**: Enter the URL of a government scheme article or upload a `.txt` file with multiple URLs.
- **Interact and Query**: After processing URLs, use the text input field to ask specific questions about the schemes. The system will respond based on the content it has processed.
- **View Embeddings and Summaries**: Check the sidebar and main display area for results and summaries.

