# LangGraph 관련 라이브러리
from langgraph.graph import StateGraph, START, END
from operator import add
from typing import Annotated, Optional
from datetime import datetime
from typing_extensions import TypedDict
from typing import Dict, Any, Literal, Union, List
from langgraph.types import Command

# LangChain 관련 라이브러리
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.document_loaders import TextLoader
from pydantic import BaseModel, Field, AnyUrl, field_validator
from dotenv import load_dotenv
from enum import Enum
import os, json, re

# gpt api 키 로드
load_dotenv()
# gpt 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini")


# 메일의 상태를 관리할 클래스
class MailState(TypedDict):
    mail_text: str # 그래프의 입력으로 들어오는 원문
    
    purpose: str
    mail_type: str
    conf_name: str
    is_cfp: bool
    start_date: str
    sub_deadline: str
    conf_website: str

# 메일 타입 열거형 정의
class MailTypeEnum(str, Enum):
    CFP_CONF = "Call for Paper for Conference"
    CFP_WORK = "Call for Paper for Workshop"
    CFP_JOUR = "Call for Paper for Journal(Issue)"
    CALL_APP = "Call for Application/Participation"
    CALL_PROP = "Call for Proposal"
    INFO = "Giving Information"
    ETC = "etc"
    
NULL_STRINGS = {"null", "none", "n/a", "na", "tbd", "unknown", ""}
    
# 추출 스키마
class Extract(BaseModel):
    conference_name: Optional[str] = Field(
        None, description='Format "<ACRONYM> <YEAR>" (e.g., "SDS 2025")'
    )
    start_date: Optional[str] = Field(
        None, description='YYYY-MM-DD or null'
    )
    submission_deadline: Optional[str] = Field(
        None, description='YYYY-MM-DD or null'
    )
    conference_website: Optional[AnyUrl] = Field(
        None, description='Official site URL or null'
    )

    # 공통: "null"/"None"/"N/A" 같은 문자열을 None으로 치환
    @field_validator("conference_name", "start_date", "submission_deadline", "conference_website", mode="before")
    @classmethod
    def _coerce_nulls(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        if s.lower() in NULL_STRINGS:
            return None
        return v
    
    @field_validator("start_date", "submission_deadline")
    def _validate_date(cls, v):
        if v is None:
            return v
        # YYYY-MM-DD 포맷 강제
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date must be YYYY-MM-DD or null")
        return v
    
    # 스킴 자동 보정 + 잡다한 꼬리문자 제거
    @field_validator("conference_website", mode="before")
    def _normalize_url(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        if not v:
            return None
        # trailing punctuation 제거 (문장 끝에 붙는 마침표, 괄호 등)
        v = v.rstrip(").,; ")
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", v):
            v = "https://" + v
        return v

# 메일 목적을 담을 클래스    
class Summary(BaseModel):
    purpose: str = Field(description="One senetence purpose of the email")

class MailTypeOut(BaseModel):
    mail_type: MailTypeEnum = Field(description="Choose exactly one label from enum.")

# 메일 요약을 위한 프롬프트와 체인
summ_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful academic-email summarizer. Output JSON."),
        ("human",
         "Summarzie the following email in exactly one sentence.\n"
         "=== EMAIL START ===\n"
         "{mail_text}"
         "=== EMAIL END ===\n"
         "Return JSON with key: purpose")
    ])
summ_chain = summ_prompt | llm | PydanticOutputParser(pydantic_object=Summary)

# 메일 타입을 분류하기 위한 프롬프트와 체인
type_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Classify the mail type. Return JSON that conforms to the provided schema.\n"
     "{format_instructions}\n"
     "PRIORITY RULES (decide using ONLY the PURPOSE string below):\n"
     "1) If the mail includes ANY invitation to submit papers/manuscripts for a CONFERENCE or WORKSHOP, "
     "   the mail_type MUST be one of:\n"
     "   - 'Call for Paper for Conference'  (for conference papers)\n"
     "   - 'Call for Paper for Workshop'    (for workshop/satellite papers)\n"
     "   Ignore other concurrent items (proposals, applications, info) if a CfP for conf/workshop exists.\n"
     "2) If both conference CfP and workshop CfP appear, choose the more SPECIFIC target:\n"
     "   - If the text explicitly invites 'workshop papers' (or a named workshop), choose 'Call for Paper for Workshop'.\n"
     "   - Otherwise choose 'Call for Paper for Conference'.\n"
     "3) If NO conf/workshop paper submission is invited but it invites proposals (workshop/tutorial/session/project), choose 'Call for Proposal'.\n"
     "4) If it invites applications/participation/registration (but not papers), choose 'Call for Application/Participation'.\n"
     "5) If it merely announces information and does not invite action, choose 'Giving Information'.\n"
     "6) When unsure, choose 'Giving Information'.\n"
     "Notes:\n"
     "- Words/phrases indicating CfP include: 'call for papers', 'paper submission', 'manuscript submission', 'regular papers', 'short papers', "
     "'workshop papers', 'camera-ready', 'submission deadline'.\n"
     "- Do NOT classify as 'Call for Paper for Journal(Issue)' unless it clearly targets a journal/special issue."
    ),
    ("human", "PURPOSE: {purpose}")
])
type_chain = type_prompt | llm | PydanticOutputParser(pydantic_object=MailTypeOut)


# 정보를 추출하기 위한 프롬프트
ext_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract the following information"
     "Return strict JSON following the schema. If any field is unknown, use null."
     "Rules:\n"
     "- conference_name: Return '<ACRONYM> <YEAR>' (e.g., 'SDS 2025'). "
     "  If the email reveals the official acronym and year (from title/body), format it like 'ICSME 2025'. "
     "  If you cannot derive a valid acronym+year, set null (do NOT invent).\n"
     "- start_date: first day of the event, 'YYYY-MM-DD' or null.\n"
     "- submission_deadline: paper submission deadline, 'YYYY-MM-DD' or null.\n"
     "- conference_website: official website URL or null.\n"
     "- No hallucination. Prefer explicit dates/URLs in the email. If partial (e.g., 'September 2025'), set null."
     "- Use actual JSON null values; never output the string 'null'."),
     ("human", "{mail_text}")
])
ext_chain = ext_prompt | llm | PydanticOutputParser(pydantic_object=Extract)

# 메일을 한 문장으로 요약하는 노드
def summarize(state: MailState) -> dict:
    mail_text = state.get("mail_text", "").strip()
    summary: Summary = summ_chain.invoke({
        "mail_text": mail_text
    })
    return {
        "purpose": summary.purpose
    }

# 메일 타입을 정하는 노드
def mail_type_node(state: MailState) -> dict:
    purpose = state.get("purpose", "")
    mail_type = type_chain.invoke({
        "format_instructions": PydanticOutputParser(pydantic_object=MailTypeOut).get_format_instructions(),
        "purpose": purpose
    })
    return {
        "mail_type": mail_type.mail_type.value
    }
    
def ext_info(state: MailState) -> dict:
    mail_text = state.get("mail_text")
    infos = ext_chain.invoke({
        "mail_text": mail_text
    })
    return {
        "conf_name": infos.conference_name,
        "start_date": infos.start_date,
        "sub_deadline": infos.submission_deadline,
        "conf_website": infos.conference_website
    }

# 라우팅 함수
def mail_router(state: MailState) -> dict:
    return state["mail_type"]
    
graph = StateGraph(MailState)
graph.add_node("summarize", summarize)
graph.add_node("mail_type_node", mail_type_node)
graph.add_node("ext_info", ext_info)

graph.add_edge(START, "summarize")
graph.add_edge("summarize", "mail_type_node")
graph.add_conditional_edges(
    "mail_type_node",
    mail_router,
    {
        "Call for Paper for Conference": "ext_info",
        "Call for Paper for Workshop": "ext_info",
        "Call for Paper for Journal(Issue)": END,
        "Call for Application/Participation": END,
        "Call for Proposal": END,
        "Giving Information": END,
        "etc": END
    })


app = graph.compile()


loader = TextLoader("./2.txt", autodetect_encoding=True)
data = loader.load()
result = app.invoke({
    "mail_text": data[0].page_content,
    "conf_name": None,
    "start_date": None,
    "sub_deadline": None,
    "conf_website": None
})

print(f"Purpose: {result['purpose']}")
print(f"Mail Type: {result['mail_type']}")
print(f"Conference Name: {result['conf_name']}")
print(f"Start Date: {result['start_date']}")
print(f"Submission Deadline: {result['sub_deadline']}")
print(f"Conference Website: {result['conf_website']}")