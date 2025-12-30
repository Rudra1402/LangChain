from typing import List

from pydantic import BaseModel, Field

class Source(BaseModel):
    """Schema for source URL used by the Agent"""

    url:str = Field(description="Source URL used by the Agent")

class AgentResponse(BaseModel):
    """Schema for Agents response with answers and sources"""

    answer:str = Field(description="Agents response to the query")
    sources:List[Source] = Field(description="List of sources used by the agent", default_factory=List)