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
llm = ChatOpenAI(model="gpt-4o-mini")

NULL_STRINGS = {"null", "none", "n/a", "na", "tbd", "unknown", ""}
LABELS = ["cfp_conf","cfp_work","cfp_jour","call_app","call_prop","info","etc"]




# 메일의 상태를 관리할 클래스
class MailState(TypedDict):
    mail_text: str  # 그래프의 입력으로 들어오는 원문

    # 요약
    purpose: str
    
    # 메일 타입에 대한 초기 판정
    mail_type_A: dict[str,bool]
    
    is_cfp: int
    not_cfp: int
    final_cfp: bool
    
    # 요약 근거 문장들
    evidence_sentences: List[str]

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



# 메일 타입을 검증하기 위해 사용할 클래스
class CFPInspectOutput(BaseModel):
    agree: bool = Field(description="Whether you agree with the preliminary is_cfp decision")
    is_cfp: bool = Field(description="Your own decision: is this email a CfP for conference/workshop papers?")






# 메일 요약을 위한 프롬프트와 체인
summ_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful academic-email summarizer. Output strict JSON matching the schema. "
     "Return: purpose (exactly one sentence), and evidence (short sentences copied verbatim from the email) "
     "that directly justify the purpose. "
     "Constraints for evidence:\n"
     "- Copy text verbatim from the email\n"
    ),
    ("human",
     "Summarize the following email in exactly one sentence (purpose), and extract evidence sentences.\n"
     "=== EMAIL START ===\n{mail_text}\n=== EMAIL END ===\n"
     "Return JSON with keys: purpose, evidence")
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
    out: Summary = summ_chain.invoke({"mail_text": mail_text})

    ev = out.evidence or []
    return {
        "purpose": out.purpose,
        "evidence_sentences": ev,
    }



# 메일 타입을 정하는 노드
def classify_flags(state: MailState) -> dict:
    purpose = state.get("purpose", "")
    
    is_cfp_count = 0
    is_not_cfp_count = 0
    
    for i in range(3):  
        flags: MailTypeFlags = flags_chain.invoke({
            "format_instructions": PydanticOutputParser(
                pydantic_object=MailTypeFlags
            ).get_format_instructions(),
            "purpose": purpose
        })
        flags = {
            "cfp_conf": flags.cfp_conf,
            "cfp_work": flags.cfp_work,
            "cfp_jour": flags.cfp_jour,
            "call_app": flags.call_app,
            "call_prop": flags.call_prop,
            "info": flags.info,
            "etc": flags.etc
        }
        if flags.get("cfp_conf") or flags.get("cfp_work"):
            is_cfp_count += 1
        else:
            is_not_cfp_count += 1
            
    return {
        "is_cfp": is_cfp_count,
        "not_cfp": is_not_cfp_count,
        "final_cfp": is_cfp_count > is_not_cfp_count,
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
    

def flags_router(state: MailState) -> str:
    return "cfp" if state.get("final_cfp", False) else "no"
    
    
    
    
graph = StateGraph(MailState)
graph.add_node("summarize", summarize)
graph.add_node("classify_flags", classify_flags)  # A
graph.add_node("ext_info", ext_info)

graph.add_edge(START, "summarize")
graph.add_edge("summarize", "classify_flags")
graph.add_conditional_edges(
    "classify_flags",
    flags_router,
    {
        "cfp": "ext_info",
        "no": END
    }
)
graph.add_edge("ext_info", END)

app = graph.compile()

def build_init_state(mail_text: str) -> dict:
    return {
        "mail_text": mail_text,
        "purpose": None,
        "mail_type_A": {k: None for k in LABELS},  # ← None으로
        "is_cfp": None,
        "not_cfp": None,
        "final_cfp": None,
        "evidence_sentences": None,
        "conf_name": None,
        "start_date": None,
        "sub_deadline": None,
        "conf_website": None
    }
    
def normalize_output(result: dict, keep_misspelled_key: bool = True) -> dict:
    """
    요청 포맷의 키로 맞춰 결과 JSON을 생성.
    keep_misspelled_key=True 이면 'coference_name' (오타) 키를 사용.
    False로 바꾸면 'conference_name'으로 저장.
    """
    key_conf_name = "coference_name" if keep_misspelled_key else "conference_name"

    return {
        "purpose": result.get("purpose"),
        "is_cfp": result.get("is_cfp"),
        "not_cfp": result.get("not_cfp"),
        "final_cfp": result.get("final_cfp"),
        key_conf_name: result.get("conf_name"),                 # 오타 키에 매핑
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





version = 2
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