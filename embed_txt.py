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
