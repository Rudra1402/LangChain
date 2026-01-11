import os
import ssl
import asyncio
import certifi

from dotenv import load_dotenv
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from utils.logger import Colors, logInfo, logSuccess, logWarning, logError, logHeader

load_dotenv()

sslContext = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512,
    chunk_size=50,
    show_progress_bar=True,
    retry_min_seconds=10
)

vectorStore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"],
    embedding=embeddings
)

tavilyExtract = TavilyExtract()
tavilyMap = TavilyMap(max_depth=5, max_breadth=20, max_pages=100)
tavilyCrawl = TavilyCrawl()

async def indexBatchDocs(docs: List[Document], batchSize: int):
    docsLen = len(docs)
    totalBatches = (docsLen // batchSize) + 1
    for i in range(0, docsLen, batchSize):
        batch = docs[i: i + batchSize]
        batchNum = (i // batchSize) + 1

        logInfo(
            f"[INFO][Pinecone] Indexing batch {batchNum}/{totalBatches} ({len(batch)} documents)",
            Colors.PURPLE
        )

        try:
            vectorStore.add_documents(batch)
            logSuccess(
                f"[SUCCESS][Pinecone] Batch {batchNum}/{totalBatches} indexed successfully!"
            )
        except Exception as e:
            logError(f"[ERROR][Pinecone] Failed to index batch {batchNum}: {str(e)}")
            raise


async def main():
    """Main async function to orchestrate the entire process"""
    logHeader("Welcome to Documentation Assitant!")
    logInfo(
        "[INFO][TavilyCrawl] Starting to Crawl documentation: https://docs.stripe.com/api",
        Colors.PURPLE
    )

    crawlResponse = tavilyCrawl.invoke({
        "url": "https://docs.stripe.com/api",
        "max_depth": 1,
        "extract_depth": "advanced"
    })

    crawlDocs = [
        Document(
            page_content=res["raw_content"],
            metadata={"source": res['url']}
        )
        for res in crawlResponse["results"]
    ]
    logSuccess(f"[SUCCESS][TavilyCrawl] Successfully crawled through {len(crawlDocs)} docs!")

    textSplitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splittedText = textSplitter.split_documents(crawlDocs)
    logSuccess(f"[SUCCESS][TavilyCrawl] Successfully splitted data into {len(splittedText)} chunks!")

    logInfo("[INFO][Pinecone] Starting batch indexing to Pinecone...", Colors.PURPLE)
    await indexBatchDocs(splittedText, 30)
    logSuccess(f"[SUCCESS][Pinecone] All {len(splittedText)} chunks indexed successfully!")


if __name__ == "__main__":
    asyncio.run(main())