from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser, StrOutputParser
from typing import TypedDict, Dict, Any

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
)

class MailState(TypedDict):
    
    free_summary: Dict[str, Any]      # LLM이 자유롭게 요약한 결과
    aligned_summary: Dict[str, Any]   # 템플릿에 맞춰 추출된 결과
    merged_summary: Dict[str, Any]    # 최종 병합 결과

def pydantic_parser_chain(prompt: ChatPromptTemplate, pmodel: type[BaseModel]):
    return prompt | llm | PydanticOutputParser(pydantic_object=pmodel)

def json_parser_chain(prompt: ChatPromptTemplate):
    return prompt | llm | JsonOutputParser()

def str_parser_chain(prompt: ChatPromptTemplate):
    return prompt | llm | StrOutputParser()