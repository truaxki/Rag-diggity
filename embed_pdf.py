import os
import shutil
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
print("Loading environment variables...")
load_dotenv()

# Define relative paths based on the location of this script
print("Defining paths...")
base_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(base_dir, 'importPDF', 'new')
processed_path = os.path.join(base_dir, 'importPDF', 'processed')
vectorstore_path = os.path.join(base_dir, 'vectorStore')

# Ensure the 'processed' and 'vectorStore' directories exist
print("Creating necessary directories if they don't exist...")
os.makedirs(processed_path, exist_ok=True)
os.makedirs(vectorstore_path, exist_ok=True)

# Initialize the PDF parser
print("Initializing the PDF parser...")
pdf_parser = LlamaParse(result_type="markdown")

# Load PDF documents from the 'new' directory
print("Loading documents from the 'new' directory...")
pdf_loader = DirectoryLoader(path=docs_path, loader_cls=TextLoader, recursive=True)
docs = []

for file_path in pdf_loader.load():
    if file_path.endswith('.pdf'):
        parsed_docs = pdf_parser.load(file_path)
        docs.extend(parsed_docs)

# Split documents into chunks for processing
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=675, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Initialize the embedding function using Google Generative AI
print("Initializing the embedding function using Google Generative AI...")
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create and persist the vectorstore using Chroma
print("Creating and persisting the vectorstore using Chroma...")
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embedding_function,
                                    persist_directory=vectorstore_path)
# Note: Persistence happens automatically with Chroma
print("Documents have been embedded and saved to ChromaDB.")

# Move processed files from 'new' to 'processed' directory
print("Moving processed files to the 'processed' directory...")
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

print("All tasks completed successfully.")
