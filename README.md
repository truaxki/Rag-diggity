
# RafTime

RafTime is a Retrieval-Augmented Generation (RAG) system designed to process and embed documents, then retrieve and generate answers to questions using Google Generative AI. It utilizes Chroma for vector storage and Langchain for seamless integration.

## Features

- **Document Embedding**: Embed text documents into a vector store for efficient retrieval.
- **Question Answering**: Retrieve relevant documents and generate answers using Google Generative AI.
- **Automated File Management**: Move processed files to designated folders.

## Directory Structure

```
RafTime/
├── embed_txt.py
├── retrieve_answer.py
├── importTXT/
│   ├── new/
│   ├── processed/
├── vectorStore/
├── .env
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` (Python package installer)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/RafTime.git
    cd RafTime
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts ctivate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory of the project.
    - Add your environment variables (e.g., API keys) to the `.env` file.

### Usage

1. **Embed Documents**:
    - Place your text files in the `importTXT/new/` directory.
    - Run the `embed_txt.py` script to process and embed the documents.
    ```bash
    python embed_txt.py
    ```

2. **Retrieve Answers**:
    - Use the `retrieve_answer.py` script to ask questions and get answers based on the embedded documents.
    ```bash
    python retrieve_answer.py
    ```

### Example

#### Embedding Documents

```python
# embed_txt.py

import os
import shutil
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load environment variables from .env file
load_dotenv()

# Define relative paths based on the location of this script
base_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(base_dir, 'importTXT', 'new')
processed_path = os.path.join(base_dir, 'importTXT', 'processed')
vectorstore_path = os.path.join(base_dir, 'vectorStore')

# Ensure the 'processed' and 'vectorStore' directories exist
os.makedirs(processed_path, exist_ok=True)
os.makedirs(vectorstore_path, exist_ok=True)

#### INDEXING ####

# Load documents from the 'new' directory
loader = DirectoryLoader(path=docs_path, loader_cls=TextLoader, recursive=True)
docs = [doc for doc in loader.load() if doc is not None]

# Split documents into chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=675, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Initialize the embedding function using Google Generative AI
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create and persist the vectorstore using Chroma
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embedding_function,
                                    persist_directory=vectorstore_path)
# Note: Persistence happens automatically with Chroma

print("Documents have been embedded and saved to ChromaDB.")

# Move processed files from 'new' to 'processed' directory
for doc in docs:
    # Get the original file path from the document's metadata
    original_path = doc.metadata['source']
    # Extract the filename from the original path
    filename = os.path.basename(original_path)
    # Define the new path in the 'processed' directory
    new_path = os.path.join(processed_path, filename)
    # Move the file to the 'processed' directory
    shutil.move(original_path, new_path)
    print(f"Moved {filename} to {processed_path}")
```

#### Retrieving Answers

```python
# retrieve_answer.py

import os
from dotenv import load_dotenv
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Define relative path based on the location of this script
base_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_path = os.path.join(base_dir, 'vectorStore')

#### RETRIEVAL ####

# Embedding function
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load vectorstore
vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_function)

# Create retriever
retriever = vectorstore.as_retriever()

# Load prompt template from Langchain hub
prompt = hub.pull("rlm/rag-prompt")

# Initialize the language model
llm = GoogleGenerativeAI(model="gemini-pro")

# Post-processing function to format documents
def format_docs(docs):
    return "

".join(doc.page_content for doc in docs)

# Create RAG (Retrieval-Augmented Generation) chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example question
question = "Who was the first female United States Naval Academy graduate to be promoted to the rank of admiral and where do I find more information on her?"

# Get answer using the RAG chain
answer = rag_chain.invoke(question)

# Print the answer
print(answer)
```

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

### License

This project is licensed under the MIT License.

### Acknowledgements

- [Langchain](https://github.com/langchain-ai/langchain)
- [Chroma](https://github.com/chroma-core/chroma)
- [Google Generative AI](https://cloud.google.com/generative-ai)

Feel free to reach out if you have any questions or suggestions!

---

Enjoy using **RagTime** to streamline your document processing and question answering!
