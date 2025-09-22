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
# LABELS = ["cfp_conf","cfp_work","cfp_jour","call_app","call_prop","info","etc"
SENT_SPLIT = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z0-9])'
    r'|\n{2,}'
)

# 중복 제거 리듀서
def merge_unique_by_raw(existing: List[Dict[str, Any]],
                        new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """conf_name_candidates 전용: raw 기준 중복 제거 병합"""
    if not existing:
        existing = []
    if not new:
        return existing

    out = list(existing)
    seen = {c.get("raw") for c in existing if c.get("raw")}

    for c in new:
        key = c.get("raw")
        if key and key not in seen:
            out.append(c)
            seen.add(key)
    return out


# 문장 분할 메소드
def split_sentences(text: str) -> list[str]:
    t = re.sub(r'[ \t]+', ' ', text).strip()
    parts = re.split(SENT_SPLIT, t)
    return [s.strip() for s in parts if s.strip()]

def build_indexed(sentences: list[str]) -> str:
    return "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))

    
class ConfNameCandidate(TypedDict, total=False):
    raw: str
    acronym: Optional[str]
    year: Optional[int]
    evidence: Optional[str]

# 메일의 상태를 관리할 클래스
class MailState(TypedDict):
    mail_text: str  # 그래프의 입력으로 들어오는 원문
    len_mail_text: int

    # 요약
    purpose: str
    len_purpose: int
    
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
    
    is_joint_conf: bool
    is_joint_work: bool

    # 추출 필드
    conf_name_candidates: Annotated[List[ConfNameCandidate], merge_unique_by_raw]  # ✅ 리스트 누적
    conf_name_final: Optional[str]
    conf_tokens: Annotated[List[str], add]
    
    # 필요한 정보가 있을 것으로 추정되는 문장들을 추출하기 위한 용도
    infos: Annotated[List[str], add]      # ✅ 문장들을 누적 저장
    infos_text: Optional[str] 
    len_infos_text: int
    
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
    
    
# 학회 이름 후보군 담을 클래스
class NameCandidate(BaseModel):
    raw: str = Field(..., description="as appears in text, e.g., 'VMCAI 2026' or 'International Conference on ...'")
    acronym: Optional[str] = Field(None, description="UPPERCASE acronym if any, e.g., 'VMCAI'")
    year: Optional[int] = Field(None, description="4-digit year if any, e.g., 2026")
    evidence: str = Field(..., description="short snippet copied verbatim around the mention")


class ExtractName(BaseModel):
    conference_name_candidates: List[NameCandidate] = Field(default_factory=list)



# 메일 요약을 위한 프롬프트와 체인
summ_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful academic-email summarizer. Output strict JSON matching the schema. "
     "Return: purpose (1 to 10 sentences, depending on the contents of the mail), and evidence (short sentences copied verbatim from the email) "
     "that directly justify the purpose. "
     "Constraints for evidence:\n"
     "- Copy text verbatim from the email\n"
    ),
    ("human",
     "Summarize the following email in 1 to 10 sentences (purpose), and extract evidence sentences.\n"
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
     "- Referring to 'Track', 'Submission', 'Paper'."
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

# Joint Conf 여부를 확인하기 위한 프롬프트
is_joint_conf_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier.\n"
     "Task: Decide if this email is about Joint Call"
     "If the expression 'Joint Conference' appears directly, return {{\"value\": true}}.\n"
     "If you check the expression 'Joint Call', make sure it's about the conference, and only if it's certain, return {{\"value\": true}}.\n"
     "Otherwise, return {{\"value\": false}}.\n"
     "Output ONLY strict JSON with key 'value'."),
    ("human",
     "{purpose}")
])
is_joint_conf_chain = is_joint_conf_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)

# Joint workshop 여부를 확인하기 위한 프롬프트
is_joint_work_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier.\n"
     "Task: Decide if this email is about Joint Call"
     "If the expression 'Joint Workshop' appears directly, return {{\"value\": true}}.\n"
     "If you check the expression 'Joint Call', make sure it's about the workshop, and only if it's certain, return {{\"value\": true}}.\n"
     "Otherwise, return {{\"value\": false}}.\n"
     "Output ONLY strict JSON with key 'value'."),
    ("human",
     "{purpose}")
])
is_joint_work_chain = is_joint_work_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)

# 학회 이름을 추출하기 위한 프롬프트
ext_name_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an information extraction model.\n"
     "Task: From the email, extract ALL names of EVENTS only: conferences or workshops.\n"
     "STRICTLY EXCLUDE journals, journal series, publishers, and venues like PACM, TOPLAS, HCI journal names, ACM DL pages, etc.\n"
     "And exclude the names of conferences/workshops referred to as co-located\n"
     "Rules:\n"
     "- Prefer '<ACRONYM> <YEAR>' if present (e.g., 'EICS 2026').\n"
     "- If a long-form event name appears on the same line (or neighboring line) with an UPPERCASE acronym and a year, set acronym to that UPPERCASE token.\n"
     "- evidence MUST be copied verbatim from near the mention.\n"
     "- Return STRICT JSON for the schema:\n{schema}"),
    ("human", "EMAIL:\n{mail_text}\n\nReturn ONLY JSON.")
]).partial(schema=ExtractName.model_json_schema())
ext_name_chain = ext_name_prompt | llm | PydanticOutputParser(pydantic_object=ExtractName)


# 한 문장씩 읽으면서 필드값이 존재하는 문장인지 판단
info_flag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier.\n"
     "Return STRICT JSON {{\"value\": true|false}}.\n"
     "true if the sentence contains ANY of the following:\n"
     "- conference/host event name (e.g., '<ACRONYM> <YEAR>', long-form names)\n"
     "- workshop name\n"
     "- start date of conference/workshop (YYYY-MM-DD, 'Nov 24, 2025', '24–28 November 2025', etc.)\n"
     "- submission deadline for papers (abstract/full/rounds)\n"
     "- official URL (http/https)\n"
     "Otherwise false. Do not explain."),
    ("human", "SENTENCE:\n{sentence}\n\nReturn ONLY JSON.")
])
info_flag_chain = info_flag_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)



# 메일을 한 문장으로 요약하는 노드
def summarize(state: MailState) -> dict:
    mail_text = state.get("mail_text", "").strip()
    out: Summary = summ_chain.invoke({"mail_text": mail_text})

    ev = out.evidence or []
    return {
        "purpose": out.purpose,
        "evidence_sentences": ev,
        "len_mail_text": len(mail_text),
        "len_purpose": len(out.purpose)
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
    infos_text = state["infos_text"]
    res: ExtractName = ext_name_chain.invoke({"mail_text": infos_text})
    cands = [c.model_dump() for c in res.conference_name_candidates]
    
    return {
        "conf_name_candidates": cands
    }

# 학회에 대한 joint call인지 확인하는 노드 
def is_joint_conf_node(state: MailState) -> dict:
    purpose = state.get("purpose", "")
    is_joint_conf = is_joint_conf_chain.invoke({"purpose": purpose}).value
    
    return {
        "is_joint_conf": is_joint_conf
    }

# 워크숍에 대한 joint call인지 확인하는 노드
def is_joint_work_node(state: MailState) -> dict:
    purpose = state.get("purpose", "")
    is_joint_work = is_joint_work_chain.invoke({"purpose": purpose}).value
    
    return {
        "is_joint_work": is_joint_work
    }
    

def build_conf_tokens_node(state: MailState) -> dict:
    """
    conf_name_candidates 중 acronym과 year가 모두 있는 항목을 골라
    'ACRONYM YEAR' 문자열로 만들어 중복 제거 후 리스트로 반환.
    """
    tokens_set = set()

    for c in state.get("conf_name_candidates", []):
        acr = c.get("acronym")
        yr  = c.get("year")
        if acr and yr:  # 둘 다 존재할 때만
            token = f"{str(acr)} {int(yr)}"
            tokens_set.add(token)

    # 리스트로 변환하여 상태 업데이트
    return {"conf_tokens": sorted(tokens_set)}



def harvest_infos_node(state: MailState) -> dict:
    mail_text = state["mail_text"]
    sentences = split_sentences(mail_text)

    picked: List[str] = []
    for s in sentences:
        # (선택) 너무 긴 문장은 LLM 안정성을 위해 잘라서 판단
        probe = s[:1000]
        flag = info_flag_chain.invoke({"sentence": probe}).value
        if flag:
            picked.append(s)

    # add 리듀서가 리스트 병합하므로 리스트로 반환
    return {"infos": picked}



# infos 문장을 하나의 문자열로 합치기
def finalize_infos_text_node(state: MailState) -> dict:
    infos_list = state.get("infos", [])
    # 순서를 유지한 채 공백으로 이어붙임 (필요하면 줄바꿈 사용)
    infos_text = "\n".join(infos_list)
    print('원문 길이: ', len(state.get("mail_text")))
    print('추출 문장 길이: ', len(infos_text))
    return {
        "infos_text": infos_text,
        "len_infos_text": len(infos_text)
    }




def cfp_candidate_router(state: MailState) -> str:
    return "go_text" if state.get("cfp_candidate", False) else "end"

def check_mail_body_router(state: MailState) -> str:
    return "go_next" if state.get("has_body", True) else "end"

def classify_cfp_router(state: MailState) -> str:
    return "go_next" if state.get("is_cfp_purpose", False) else "end"

def is_joint_conf_router(state: MailState) -> str:
    return "end" if state.get("is_joint_conf", False) else "go_next"
    
    
    
graph = StateGraph(MailState)
graph.add_node("summarize", summarize)
graph.add_node("cfp_candidate", cfp_candidate_node)
graph.add_node("classify_cfp_purpose", classify_cfp_purpose_node)
graph.add_node("check_mail_body", check_mail_body_node)
graph.add_node("ext_name", ext_name_node)
graph.add_node("is_joint_conf", is_joint_conf_node)
graph.add_node("is_joint_work", is_joint_work_node)
graph.add_node("build_conf_tokens", build_conf_tokens_node)
graph.add_node("harvest_infos", harvest_infos_node)
graph.add_node("finalize_infos_text", finalize_infos_text_node)


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
        "go_next": "is_joint_conf",
        "end": END
    }
)
graph.add_conditional_edges(
    "is_joint_conf",
    is_joint_conf_router,
    {
        "go_next": "is_joint_work",
        "end": END
    }
)
graph.add_edge("is_joint_work", "harvest_infos")
graph.add_edge("harvest_infos", "finalize_infos_text")
graph.add_edge("finalize_infos_text", "ext_name")
graph.add_edge("ext_name", "build_conf_tokens")
graph.add_edge("build_conf_tokens", END)

app = graph.compile()

def build_init_state(mail_text: str) -> dict:
    return {
        "mail_text": mail_text,
        "cfp_candidate": None,
        "purpose": None,
        "classify_cfp_purpose": None,
        "is_cfp_purpose": False,
        "is_cfp": None,
        "has_body": True,
        "is_joint_conf": False,
        "is_joint_work": False,
        "infos": [],
        "infos_text": None,
        "evidence_sentences": None,
        "conf_name_candidates": [],
        "conf_name_final": None,
        "conf_tokens": [],
        "start_date": None,
        "sub_deadline": None,
        "conf_website": None,
        "len_mail_text": 0,
        "len_purpose": 0,
        "len_infos_text": 0
    }
    
def normalize_output(result: dict, keep_misspelled_key: bool = True) -> dict:
    return {
        "has_body": result.get("has_body"),
        "purpose": result.get("purpose"),
        "cfp": {
            "cfp_candidate": result.get("cfp_candidate"),
            "classify_cfp_purpose": result.get("classify_cfp_purpose"),
            "is_cfp_purpose": result.get("is_cfp_purpose"),
        },
        "is_cfp": result.get("is_cfp"),  
        "is_joint_call": {
            "is_joint_conf": result.get("is_joint_conf"),
            "is_joint_work": result.get("is_joint_work"),
        },
        "infos": {
            "infos": result.get("infos"),
            "conf_name_candidates": result.get("conf_name_candidates"),
            "conf_name_final": result.get("conf_name_final"),
            "conf_name_tokens": result.get("conf_tokens"),
            "start_date": result.get("start_date"),
            "submission_deadline": result.get("sub_deadline"),
            "conference_website": result.get("conf_website"),
            # "evidence_sentences": result.get("evidence_sentences")
        },
        "length": {
            "mail_text": result.get("len_mail_text"),
            "purpose": result.get("len_purpose"),
            "infos": result.get("len_infos_text")
        }
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



# --- 여기서 단일 파일 실행 ---
input_path = Path("./data/texts/4.txt")           # 입력 파일
output_path = Path("./single_predict.json")

out = process_one_file(input_path, keep_misspelled_key=True)
save_json(out, output_path)

print(f"[OK] {input_path.name} -> {output_path.name}")



# 단일 파일에 대해 여러번 실행
# for i in range(10):
#     input_path = Path("./data/texts/41.txt")           # 입력 파일
#     output_path = Path(f"single_{i}_predict.json")

#     out = process_one_file(input_path, keep_misspelled_key=True)
#     save_json(out, output_path)

#     print(f"[OK] {input_path.name} -> {output_path.name}")