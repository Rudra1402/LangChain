from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def addServiceTax(price:float) -> float:
    """
    Docstring for addServiceTax
    :param price: number to add 10% tax to
    :returns: number with 10% tax added
    """
    print("Calling Tool - addServiceTax")
    percent10 = price / 10
    finalPrice = percent10 + price
    finalPrice = round(finalPrice, 2)

    return finalPrice

tavilySearch = TavilySearch(max_results=1)

tools = [tavilySearch, addServiceTax]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).bind_tools(tools)