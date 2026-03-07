from dotenv import load_dotenv
from langchain.messages import HumanMessage
from graph.graphRun import app

load_dotenv()

if __name__ == "__main__":
    print("Welcome to Agentic RAG!")
    res = app.invoke({"question": "What is Agent Memory?"}) # type: ignore
    print(res)