from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class HallucinationSchema(BaseModel):
    """Binary score to check if the LLM generated response satisfies the question"""

    binaryScore:str = Field(description="Answer addresses the question, 'yes' or 'no'")

llm = ChatOpenAI(temperature=0)
structuredLLM = llm.with_structured_output(HallucinationSchema)

initialSystemMsg = """You are a grader assessing whether an answer addresses / resolves a question \n 
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
"""

chatPrompt = ChatPromptTemplate.from_messages([
    ("system", initialSystemMsg),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
])

answerGraderChain = chatPrompt | structuredLLM