from langchain.prompt import ChatPromptTemplate
from langchain_core.parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI

generate_queries = (
    prompt_perspectives
    | GoogleGenerativeAI
    | StrOutputParser()
    |(lambda x: x.split("\n")
    )


generate_queries