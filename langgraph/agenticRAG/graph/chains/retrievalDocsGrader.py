from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

class GradeDocs(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binaryScore:str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structuredLLM = llm.with_structured_output(GradeDocs)

initialSystemMessage = """You are a grader assessing relevance of a retrieved document to a user question. \n 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

chatPrompt = ChatPromptTemplate.from_messages([
    ("system", initialSystemMessage),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
])

gradeResultsChain = chatPrompt | structuredLLM