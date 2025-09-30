from __future__ import annotations
from typing import Annotated, Optional, List, Dict, Any, Literal
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum
import re, json, os
from datetime import date
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# --- LLM ---
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    seed=42
)

# --- Constants / Regex ---
BATCH_SIZE = 5
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

class ExtractText(BaseModel):
    extracted_text: str = Field(description="Extracted text related to Call for Papers")
    
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
    
class ConferenceDate(BaseModel):
    """학회 시작 날짜를 추출하기 위한 스키마"""

    start_date: Optional[date] = Field(
        default = None, 
        description="Start date of the conference or symposium")
    
class DeadlineInfo(BaseModel):
    """A single deadline entry with its context and normalized date."""
    raw_text: str = Field(description="The original sentence or phrase containing the deadline, extracted verbatim.")
    normalized_date: str = Field(description="The extracted date from the raw_text, normalized to YYYY-MM-DD format.")
    
class DeadlineCandidates(BaseModel):
    candidates: List[DeadlineInfo]
    
class SubmissionDate(BaseModel):
    """논문 제출 마감일을 추출하기 위한 스키마"""

    sub_deadline: Optional[date] = Field(
        default = None, 
        description="Submission deadline for papers (abstract/full/rounds)")
    
class ConferenceUrl(BaseModel):
    """Schema for extracting the main conference URL."""

    conf_url: Optional[str] = Field(
        default=None,
        description="The official homepage URL of the main conference. It must not be a link to a submission system or a publisher."
    )

# --- MailState ---
class MailState(TypedDict):
    mail_text: str
    len_mail_text: int
    
    extracted_text: str
    len_extracted_text: int
    
    purpose: str
    evidence_sentences: List[str]
    len_purpose: int
    
    cfp_candidate: bool
    classify_cfp_target: str
    classify_cfp_mail_text: str
    is_cfp: bool
    is_cfp_final: bool
    
    has_body: bool
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
    sub_deadline_candidate: Optional[List[str]]
    conf_website: Optional[str]

# --- Chain helpers (각 모듈에서 재사용할 프롬프트 생성기) ---
def parser_chain(prompt: ChatPromptTemplate, pmodel: type[BaseModel]):
    return prompt | llm | PydanticOutputParser(pydantic_object=pmodel)