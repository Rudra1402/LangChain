from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
subDocs = [item for subDoc in docs for item in subDoc]

txtSplitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

splittedDocs = txtSplitter.split_documents(subDocs)
embeddings = OpenAIEmbeddings()

# Needs to run only once to fetch, split, and embed website data.
# No need to embed the same data on each run (waste of tokens, memory)

# vectorStore = Chroma.from_documents(
#     documents=splittedDocs,
#     collection_name="agentic-rag",
#     embedding=embeddings,
#     persist_directory="./.chroma"
# )

retrieverStore = Chroma(
    embedding_function=embeddings,
    collection_name="agentic-rag",
    persist_directory="./.chroma"
).as_retriever()