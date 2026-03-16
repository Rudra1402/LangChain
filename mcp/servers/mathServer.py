from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import math

load_dotenv()

mcp = FastMCP("Math")

@mcp.tool()
def addNums(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiplyNums(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    print("Welcome to MCP Server!")
    mcp.run(transport="stdio")