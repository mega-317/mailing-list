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
    mail_text: str  # 그래프의 입력으로 들어오는 원문

    # 요약
    purpose: str

    # 멀티 라벨 결과 (각 타입별 bool)
    cfp_conf: bool
    cfp_work: bool
    cfp_jour: bool
    call_app: bool
    call_prop: bool
    info: bool
    etc: bool  # 아무 규칙에도 안 걸리지만 특정 분류가 필요한 경우 대비(선택)

    # 추출 필드
    conf_name: Optional[str]
    start_date: Optional[str]
    sub_deadline: Optional[str]
    conf_website: Optional[str]

# 메일 타입 열거형 정의
class MailTypeFlags(BaseModel):
    # Call for Papers
    cfp_conf: bool = Field(description="Conference paper CfP present?")
    cfp_work: bool = Field(description="Workshop paper CfP present?")
    cfp_jour: bool = Field(description="Journal/Special Issue CfP present?")

    # Call for ...
    call_app: bool = Field(description="Call for Application / Participation?")
    call_prop: bool = Field(description="Call for Proposal?")

    # Info / Etc
    info: bool = Field(description="Only information / announcements?")
    etc: bool = Field(description="Other / Unclear category?")
    
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
flags_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You classify an academic mailing-list email into multiple labels.\n"
     "Return strict JSON that conforms to the provided schema.\n"
     "{format_instructions}\n"
     "GUIDELINES:\n"
     "- Decide using ONLY the PURPOSE string provided.\n"
     "- Set each field to true/false independently (multi-label).\n"
     "- If the PURPOSE clearly invites paper submissions to a conference, set cfp_conf=true.\n"
     "- If it invites workshop paper submissions, set cfp_work=true.\n"
     "- If it invites journal/special issue submissions, set cfp_jour=true.\n"
     "- If it invites applications/participation/registration (not papers), set call_app=true.\n"
     "- If it invites proposals (workshop/tutorial/session/project), set call_prop=true.\n"
     "- If it mainly announces info and no direct action is requested, set info=true.\n"
     "- If unclear or outside categories, you may set etc=true.\n"
     "- Prefer precision: if uncertain, leave fields false rather than guessing."  # 보수적 판정
    ),
    ("human", "PURPOSE: {purpose}")
])
flags_chain = flags_prompt | llm | PydanticOutputParser(pydantic_object=MailTypeFlags)


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
def classify_flags(state: MailState) -> dict:
    purpose = state.get("purpose", "")
    flags: MailTypeFlags = flags_chain.invoke({
        "format_instructions": PydanticOutputParser(pydantic_object=MailTypeFlags).get_format_instructions(),
        "purpose": purpose
    })
    return {
        "cfp_conf": flags.cfp_conf,
        "cfp_work": flags.cfp_work,
        "cfp_jour": flags.cfp_jour,
        "call_app": flags.call_app,
        "call_prop": flags.call_prop,
        "info": flags.info,
        "etc": flags.etc
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

# 라우팅: 컨퍼런스 또는 워크숍 CfP면 추출로, 아니면 종료
def mail_router(state: MailState) -> str:
    if state.get("cfp_conf") or state.get("cfp_work"):
        return "EXT"
    return "END"
    
graph = StateGraph(MailState)
graph.add_node("summarize", summarize)
graph.add_node("classify_flags", classify_flags)
graph.add_node("ext_info", ext_info)

graph.add_edge(START, "summarize")
graph.add_edge("summarize", "classify_flags")
graph.add_conditional_edges(
    "classify_flags",
    mail_router,
    {"EXT": "ext_info", "END": END}
)


app = graph.compile()


loader = TextLoader("./2.txt", autodetect_encoding=True)
data = loader.load()

# 초기 상태: bool들은 기본 False로 시작
init_state = {
    "mail_text": data[0].page_content,
    "purpose": "",
    "cfp_conf": False,
    "cfp_work": False,
    "cfp_jour": False,
    "call_app": False,
    "call_prop": False,
    "info": False,
    "etc": False,
    "conf_name": None,
    "start_date": None,
    "sub_deadline": None,
    "conf_website": None
}
result = app.invoke(init_state)


def derive_primary_label(r):
    if r["cfp_work"]:
        return "Call for Paper for Workshop"
    if r["cfp_conf"]:
        return "Call for Paper for Conference"
    if r["cfp_jour"]:
        return "Call for Paper for Journal(Issue)"
    if r["call_prop"]:
        return "Call for Proposal"
    if r["call_app"]:
        return "Call for Application/Participation"
    if r["info"]:
        return "Giving Information"
    if r["etc"]:
        return "etc"
    return "Unlabeled"


print(f"Purpose: {result['purpose']}")
print("Flags:\n"
      f"cfp_conf={result['cfp_conf']}\n"
      f"cfp_work={result['cfp_work']}\n" 
      f"cfp_jour={result['cfp_jour']}\n"
      f"call_app={result['call_app']}\n" 
      f"call_prop={result['call_prop']}\n" 
      f"info={result['info']}\n" 
      f"etc={result['etc']}")
print(f"(Derived Primary) Mail Type: {derive_primary_label(result)}")
print(f"Conference Name: {result['conf_name']}")
print(f"Start Date: {result['start_date']}")
print(f"Submission Deadline: {result['sub_deadline']}")
print(f"Conference Website: {result['conf_website']}")