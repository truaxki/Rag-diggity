import os
from dotenv import load_dotenv
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_path = os.path.join(base_dir, 'vectorStore')

#### RETRIEVAL ###
# Embedding function
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load vectorstore
vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_function)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = GoogleGenerativeAI(model="gemini-pro")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Example question
question = "what must all navy commands with with an intelegence mission complete by 15Jul2024?"
answer = rag_chain.invoke(question)
print(answer)

docs = retriever.get_relevant_documents(question)
print(docs)