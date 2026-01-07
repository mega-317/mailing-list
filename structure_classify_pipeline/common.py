from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser, StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
)

class MailState(TypedDict):
    input_json: dict
    mail_type: bool

def pydantic_parser_chain(prompt: ChatPromptTemplate, pmodel: type[BaseModel]):
    return prompt | llm | PydanticOutputParser(pydantic_object=pmodel)

def json_parser_chain(prompt: ChatPromptTemplate):
    return prompt | llm | JsonOutputParser()

def str_parser_chain(prompt: ChatPromptTemplate):
    return prompt | llm | StrOutputParser()