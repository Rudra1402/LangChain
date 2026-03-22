from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain

load_dotenv()

llm = ChatOpenAI()

textLoader = TextLoader("/Users/rudrapatel/Desktop/Projects/langchain-agents/constants/medium.txt")
textDocs = textLoader.load()

textSplitter = RecursiveCharacterTextSplitter(["\n\n", "\n"], chunk_size=800, chunk_overlap=100)
splittedDocs = textSplitter.split_documents(textDocs)

combinedDocsContext = load_summarize_chain(llm, "refine")

if __name__ == "__main__":
    print("Context Refiner!")
    res = combinedDocsContext.run(splittedDocs)
    print(res)