from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("Weather")

@mcp.tool()
def getWeather(city: str) -> str:
    """Get weather for the given city"""
    return f"Weather in {city} is quite sunny with temperature of 30deg celsius!"

if __name__ == "__main__":
    print("Welcome to MCP Server! Weather!")
    mcp.run(transport="sse")
