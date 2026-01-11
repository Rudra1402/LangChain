import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    loader = TextLoader("/Users/rudrapatel/Desktop/Projects/langchain-agents/constants/medium.txt")
    document = loader.load()

    textSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splittedText = textSplitter.split_documents(document)

    embeddingsOpenAI = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    PineconeVectorStore.from_documents(splittedText, embeddingsOpenAI, index_name=os.environ.get("INDEX_NAME"))

    print("Welcome to RAG!", splittedText)