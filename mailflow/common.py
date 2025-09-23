from __future__ import annotations
from typing import Annotated, Optional, List, Dict, Any, Literal
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field
from enum import Enum
import re, json, os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# --- LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Constants / Regex ---
BATCH_SIZE = 10
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])|\n{2,}')
NULL_STRINGS = {"null", "none", "n/a", "na", "tbd", "unknown", ""}

# --- Reducers ---
def merge_unique_by_raw(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not existing:
        existing = []
    if not new:
        return existing
    out = list(existing)
    seen = {c.get("raw") for c in existing if c.get("raw")}
    for c in new:
        key = c.get("raw")
        if key and key not in seen:
            out.append(c); seen.add(key)
    return out

# --- Utils ---
def split_sentences(text: str) -> list[str]:
    t = re.sub(r'[ \t]+', ' ', text).strip()
    parts = re.split(SENT_SPLIT, t)
    return [s.strip() for s in parts if s.strip()]

# --- Pydantic Schemas (한 곳에서만 정의) ---
class BoolOut(BaseModel):
    value: bool = Field(description="Return true or false")

class BoolListOut(BaseModel):
    flags: List[bool] = Field(description="Same length/ order as input")

class Summary(BaseModel):
    purpose: str = Field(description="1~10문장 목적 요약")
    evidence: List[str] = Field(description="근거 문장들(메일에서 발췌)")

class CFPLabel(str, Enum):
    conference = "conference"
    workshop = "workshop"
    journal = "journal"

class CFPLabelParser(BaseModel):
    label: CFPLabel

class NameCandidate(BaseModel):
    raw: str = Field(..., description="e.g., 'VMCAI 2026' or long form")
    acronym: Optional[str] = None
    year: Optional[int] = None
    evidence: str = Field(...)

class ExtractName(BaseModel):
    Name_candidates: List[NameCandidate] = Field(default_factory=list)

class ConfChoice(BaseModel):
    choice: str = Field(description="verbatim from candidates")

# --- MailState ---
class MailState(TypedDict):
    mail_text: str
    len_mail_text: int
    purpose: str
    len_purpose: int
    cfp_candidate: bool
    classify_cfp_purpose: str
    classify_cfp_mail_text: str
    is_cfp_purpose: bool
    is_cfp: bool
    has_body: bool
    evidence_sentences: List[str]
    is_joint_conf: bool
    is_joint_work: bool
    infos: Annotated[List[str], add]
    infos_text: Optional[str]
    len_infos_text: int
    conf_name_candidates: Annotated[List[Dict[str, Any]], merge_unique_by_raw]
    conf_name_final: Optional[str]
    conf_tokens: Annotated[List[str], add]
    work_name_candidates: Annotated[List[Dict[str, Any]], merge_unique_by_raw]
    work_tokens: Annotated[List[str], add]
    start_date: Optional[str]
    sub_deadline: Optional[str]
    conf_website: Optional[str]

# --- Chain helpers (각 모듈에서 재사용할 프롬프트 생성기) ---
def parser_chain(prompt: ChatPromptTemplate, pmodel: type[BaseModel]):
    return prompt | llm | PydanticOutputParser(pydantic_object=pmodel)