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

import glob
from pathlib import Path

# gpt api 키 로드
load_dotenv()
# gpt 모델 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
    )

NULL_STRINGS = {"null", "none", "n/a", "na", "tbd", "unknown", ""}
# LABELS = ["cfp_conf","cfp_work","cfp_jour","call_app","call_prop","info","etc"]
    

# 메일의 상태를 관리할 클래스
class MailState(TypedDict):
    mail_text: str  # 그래프의 입력으로 들어오는 원문

    # 요약
    purpose: str
    
    cfp_candidate: bool
    
    classify_cfp_purpose: str
    classify_cfp_mail_text: str
    
    is_cfp_purpose: bool
    is_cfp_mail_text: bool
    
    is_cfp: bool
    
    # 메일 내용에 이상이 없는지 확인용
    has_body: bool
    
    # 요약 근거 문장들
    evidence_sentences: List[str]

    # 추출 필드
    conf_name_candidates: str
    conf_name_final: Optional[str]  # 최종 1개 (예: "VMCAI 2026")
    
    start_date: Optional[str]
    sub_deadline: Optional[str]
    conf_website: Optional[str]


class BoolOut(BaseModel):
    value: bool = Field(description="Return true or false")


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
    evidence: List[str] = Field(
        description="short evidence sentences copied verbatim from the email that justify the purpose"
    )
    
class CFPLabel(str, Enum):
    conference = "conference"
    workshop = "workshop"
    journal = "journal"
    
    
class CFPLabelParser(BaseModel):
    label: CFPLabel = Field(description="One of: conference, workshop, journal")
    
    
    
# # 학회 이름 후보군을 담을 클래스
# class NameCandidate(BaseModel):
#     raw: str = Field(description="Raw candidate text")
#     acronym: Optional[str] = None
#     year: Optional[int] = None

# class ExtractName(BaseModel):
#     conference_name_candidates: List[NameCandidate] = Field(default_factory=list)



# 메일 요약을 위한 프롬프트와 체인
summ_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful academic-email summarizer. Output strict JSON matching the schema. "
     "Return: purpose (1 to 5 sentences, depending on the contents of the mail), and evidence (short sentences copied verbatim from the email) "
     "that directly justify the purpose. "
     "Constraints for evidence:\n"
     "- Copy text verbatim from the email\n"
    ),
    ("human",
     "Summarize the following email in 1 to 5 sentences (purpose), and extract evidence sentences.\n"
     "=== EMAIL START ===\n{mail_text}\n=== EMAIL END ===\n"
     "Return JSON with keys: purpose, evidence")
])
summ_chain = summ_prompt | llm | PydanticOutputParser(pydantic_object=Summary)


# 이 메일이 CFP 메일인지 판정하는 프롬프트와 체인
cfp_candidate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier for academic emails.\n"
     "Task: I'll give you mail's purpose. Determine if the email is a Call for Papers (CFP).\n"
     "Definition of CFP:\n"
     "- Asking to submit a paper for a conference or workshop\n"
     "- Requesting a manuscript for a journal special issue\n\n"
     "Proposal mail is not applicable, such as requesting a proposal for a workshop or satellite event.\n"
     "If yes → return JSON {{\"value\": true}}\n"
     "If no → return JSON {{\"value\": false}}\n"
     "Rules:\n"
     "- Output ONLY strict JSON with key 'value'.\n"
     "- Never add explanations or extra text."),
    ("human",
     "Classify the following email based on mail's purpose:\n"
     "{purpose}")
])
cfp_candidate_chain = cfp_candidate_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)



# 무엇에 대한 cfp 메일인지 분류하는 프롬프트
classify_cfp_purpose_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict classifier for academic emails.\n"
     "Task: Read the purpose of email and decide which type of Call for Papers (CFP) it is.\n"
     "Classes (choose exactly one):\n"
     "  - conference  : a CFP for a conference\n"
     "  - workshop    : a CFP for a workshop (co-located or standalone)\n"
     "  - journal     : a CFP for an academic journal or special issue (edited books/book chapters also belong here)\n\n"
     "Assume every email you receive here is indeed a CFP (no need to reject).\n"
     "Output ONLY a strict JSON object with a single key 'label' whose value is one of: "
     "'conference', 'workshop', 'journal'.\n"
     "Do not include any explanations or extra text."
    ),
    ("human",
     "Classify the following email based on mail's purpose:\n"
     "{purpose}")
])
classify_cfp_purpose_chain = classify_cfp_purpose_prompt | llm | PydanticOutputParser(pydantic_object=CFPLabelParser)


# 메일 본문이 정상적인지 확인하는 프롬프트
check_mail_body_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier.\n"
     "Task: Decide if the email body has meaningful content.\n"
     "Rules:\n"
     "- If the body is empty or effectively empty (only headers, boilerplate, or fewer than 3 non-empty lines), return {{\"value\": false}}.\n"
     "- Otherwise, return {{\"value\": true}}.\n"
     "Output ONLY strict JSON with key 'value'."),
    ("human", "{mail_text}")
])
check_mail_body_chain = check_mail_body_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)



# 학회 이름을 추출하기 위한 프롬프트
ext_name_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Please read the text of email I give you and return the texts that are supposed to be the name of the conference."
     "If there is only one, please print out one, and if there seem to be multiple, print out multiple."
     "Return strict JSON following the schema. If any field is unknown, use null."
     "Rules:\n"
     "- Conference_name: Return '<ACRONYM> <YEAR>' (e.g., 'SDS 2025'). "
     "- If the email reveals the official acronym and year (from title/body), format it like 'ICSME 2025'. "
     "- If you cannot derive a valid acronym+year, set null (do NOT invent).\n"
     "- If there are multiple presumption names, please separate them with commas"
     ),
     ("human", "{mail_text}")
])
ext_name_chain = ext_name_prompt | llm | StrOutputParser()







# 메일을 한 문장으로 요약하는 노드
def summarize(state: MailState) -> dict:
    mail_text = state.get("mail_text", "").strip()
    out: Summary = summ_chain.invoke({"mail_text": mail_text})

    ev = out.evidence or []
    return {
        "purpose": out.purpose,
        "evidence_sentences": ev,
    }


# 이 메일이 cfp 인지 분류하는 노드
def cfp_candidate_node(state: MailState) -> dict:
    purpose = state.get("purpose", "")
    is_cfp = cfp_candidate_chain.invoke({"purpose": purpose}).value
    
    return {
        "cfp_candidate": is_cfp
    }

# 이 메일이 무엇에 대한 cfp인지 분류하는 노드
def classify_cfp_purpose_node(state: MailState) -> dict:
    purpose = state.get("purpose", "")
    classify_cfp_purpose = classify_cfp_purpose_chain.invoke({"purpose": purpose}).label
    
    if classify_cfp_purpose == "conference" or classify_cfp_purpose == "workshop":
        is_cfp_purpose = True
    else:
        is_cfp_purpose = False
    
    return {
        "classify_cfp_purpose": classify_cfp_purpose,
        "is_cfp_purpose": is_cfp_purpose
    }
    
    
# 메일 본문이 제대로 존재하는지 확인하는 노드
def check_mail_body_node(state: MailState) -> dict:
    mail_text = state.get("mail_text", "")
    line_count = len(mail_text.strip().splitlines())
    
    if line_count < 100:
        result = check_mail_body_chain.invoke({"mail_text": mail_text}).value
        
        return {
            "has_body": result
        }


# 학회/워크숍 이름을 추출하는 노드
def ext_name_node(state: MailState) -> dict:
    mail_text = state.get("mail_text")
    conf_name_candidates = ext_name_chain.invoke({
        "mail_text": mail_text
    })
    return {
        "conf_name_candidates": conf_name_candidates
    }
    

def cfp_candidate_router(state: MailState) -> str:
    return "go_text" if state.get("cfp_candidate", False) else "end"

def check_mail_body_router(state: MailState) -> str:
    return "go_next" if state.get("has_body", True) else "end"

def classify_cfp_router(state: MailState) -> str:
    return "go_next" if state.get("is_cfp_purpose", False) else "end"
    
    
    
graph = StateGraph(MailState)
graph.add_node("summarize", summarize)
graph.add_node("cfp_candidate", cfp_candidate_node)
graph.add_node("classify_cfp_purpose", classify_cfp_purpose_node)
graph.add_node("check_mail_body", check_mail_body_node)
graph.add_node("ext_name", ext_name_node)


graph.add_edge(START, "check_mail_body")
graph.add_conditional_edges(
    "check_mail_body",
    check_mail_body_router,
    {
        "go_next": "summarize",
        "end": END
    }
)
graph.add_edge("summarize", "cfp_candidate")
graph.add_conditional_edges(
    "cfp_candidate",
    cfp_candidate_router,
    {
        "go_text": "classify_cfp_purpose",  # 노드 이름
        "end": END
    }
)
graph.add_conditional_edges(
    "classify_cfp_purpose",
    classify_cfp_router,
    {
        "go_next": "ext_name",
        "end": END
    }
)
graph.add_edge("ext_name", END)

app = graph.compile()

def build_init_state(mail_text: str) -> dict:
    return {
        "mail_text": mail_text,
        "cfp_candidate": None,
        "purpose": None,
        "classify_cfp_purpose": None,
        "classfiy_cfp_mail_text": None,
        "is_cfp_purpose": False,
        "is_cfp_mail_text": None,
        "is_cfp": None,
        "has_body": True,
        "evidence_sentences": None,
        "conf_name_candidates": None,
        "conf_name_final": None,
        "start_date": None,
        "sub_deadline": None,
        "conf_website": None
    }
    
def normalize_output(result: dict, keep_misspelled_key: bool = True) -> dict:
    return {
        "has_body": result.get("has_body"),
        "purpose": result.get("purpose"),
        "cfp_candidate": result.get("cfp_candidate"),
        "classify_cfp_purpose": result.get("classify_cfp_purpose"),
        "classfiy_cfp_mail_text": result.get("classify_cfp_mail_text"),
        "is_cfp_purpose": result.get("is_cfp_purpose"),
        "is_cfp_mail_text": result.get("is_cfp_mail_text"),
        "is_cfp": result.get("is_cfp"),
        "conf_name_candidates": result.get("conf_name_candidates"),
        "conf_name_final": result.get("conf_name_final"),
        "start_date": result.get("start_date"),
        "submission_deadline": result.get("sub_deadline"),
        "conference_website": result.get("conf_website"),
        "evidence_sentences": result.get("evidence_sentences")
    }

def process_one_file(txt_path: Path, keep_misspelled_key: bool = True) -> dict:
    # 파일 로드
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        mail_text = f.read()

    # 상태 생성 및 실행
    state = build_init_state(mail_text)
    result = app.invoke(state)

    # 결과 정규화(요청 포맷)
    out_json = normalize_output(result, keep_misspelled_key=keep_misspelled_key)
    return out_json

def save_json(obj: dict, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)





version = 10
TEXT_DIR = Path("./data/texts")
OUT_DIR = Path(f"./data/predictions_{version}")  # 결과 저장 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)


# === 배치 실행 ===
# 파일명이 숫자.txt 형태(예: 1.txt, 2.txt, ... 100.txt)라면 아래처럼 정렬
txt_files = []
for p in glob.glob(str(TEXT_DIR / "*.txt")):
    name = Path(p).stem
    try:
        n = int(name)
    except ValueError:
        # 숫자 파일만 대상으로
        continue
    txt_files.append((n, Path(p)))

# 숫자 기준 오름차순 정렬
txt_files.sort(key=lambda x: x[0])

errors = []
for n, path in txt_files:
    try:
        out = process_one_file(path, keep_misspelled_key=True)  # 요청 포맷(오타 포함) 준수
        out_name = f"{n}_predict.json"  # 11_predict.json 같은 형태
        save_json(out, OUT_DIR / out_name)
        print(f"[OK] {path.name} -> {out_name}")
    except Exception as e:
        errors.append((path.name, str(e)))
        print(f"[FAIL] {path.name} -> {e}")

# 실패 로그 간단 출력
if errors:
    print("\n=== FAILED FILES ===")
    for name, msg in errors:
        print(f"- {name}: {msg}")