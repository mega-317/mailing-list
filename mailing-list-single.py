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
LABELS = ["cfp_conf","cfp_work","cfp_jour","call_app","call_prop","info","etc"]




# 메일의 상태를 관리할 클래스
class MailState(TypedDict):
    mail_text: str  # 그래프의 입력으로 들어오는 원문

    # 요약
    purpose: str
    
    is_cfp: bool
    
    cfp_conf: bool
    cfp_work: bool
    cfp_jour: bool
    call_app: bool
    call_prop: bool
    info: bool
    etc: bool
    
    is_cfp: bool
    
    # 요약 근거 문장들
    evidence_sentences: List[str]

    # 추출 필드
    conf_name: Optional[str]
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



is_cfp_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier for academic emails.\n"
     "Task: Determine if the email is a Call for Papers (CFP).\n"
     "Definition of CFP:\n"
     "- Asking to submit a paper for a conference or workshop\n"
     "- Requesting a manuscript for a journal special issue\n\n"
     "If yes → return JSON {{\"value\": true}}\n"
     "If no → return JSON {{\"value\": false}}\n"
     "Rules:\n"
     "- Output ONLY strict JSON with key 'value'.\n"
     "- Never add explanations or extra text."),
    ("human",
     "Classify the following email:\n"
     "=== EMAIL START ===\n{mail_text}\n=== EMAIL END ===")
])
is_cfp_chain = is_cfp_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)



# flag_conf_cfp_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You're going to play the role of sorting mail.\n"
#      "From now on, I will give you a text summarizing the mail in one sentence.\n"
#      "read the sentence, and return True, otherwise False\n"
#      "You should only determine that this mail is true when you are recruiting papers to submit to the conference accurately.\n"
#      "Return strict JSON: {{\"value\": <true|false>}}.\n"),
#     ("human", "PURPOSE: {purpose}")
# ])
# flag_conf_cfp_chain = flag_conf_cfp_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)

# flag_work_cfp_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You're going to play the role of sorting mail.\n"
#      "From now on, I will give you a text summarizing the mail in one sentence.\n"
#      "read the sentence, and return True, otherwise False\n"
#      "You should only determine that this mail is true when you are recruiting papers to submit to the workshop accurately.\n"
#      "Return strict JSON: {{\"value\": <true|false>}}.\n"),
#     ("human", "PURPOSE: {purpose}")
# ])
# flag_work_cfp_chain = flag_work_cfp_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)

# flag_jour_cfp_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You're going to play the role of sorting mail.\n"
#      "From now on, I will give you a text summarizing the mail in one sentence.\n"
#      "read the sentence, and return True, otherwise False\n"
#      "if you think this mail is a recruiting article for the Journal or Issues\n"
#      "Return strict JSON: {{\"value\": <true|false>}}.\n"),
#     ("human", "PURPOSE: {purpose}")
# ])
# flag_jour_cfp_chain = flag_jour_cfp_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)

# flag_call_app_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You're going to play the role of sorting mail.\n"
#      "From now on, I will give you a text summarizing the mail in one sentence.\n"
#      "read the sentence, and return True, otherwise False\n"
#      "if you think this mail is promoting participation or application for a specific program\n"
#      "Return strict JSON: {{\"value\": <true|false>}}.\n"),
#     ("human", "PURPOSE: {purpose}")
# ])
# flag_call_app_chain = flag_call_app_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)

# flag_call_prop_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You're going to play the role of sorting mail.\n"
#      "From now on, I will give you a text summarizing the mail in one sentence.\n"
#      "read the sentence, and return True, otherwise False\n"
#      "if you think this mail is asking for a proposal such as a subsidiary event\n"
#      "Return strict JSON: {{\"value\": <true|false>}}.\n"),
#     ("human", "PURPOSE: {purpose}")
# ])
# flag_call_prop_chain = flag_call_prop_prompt | llm | PydanticOutputParser(pydantic_object=BoolOut)






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
# def classify_flags(state: MailState) -> dict:
#     purpose = state.get("purpose", "")
#     cfp_conf = flag_conf_cfp_chain.invoke({"purpose": purpose}).value
#     cfp_work = flag_work_cfp_chain.invoke({"purpose": purpose}).value
#     cfp_jour = flag_jour_cfp_chain.invoke({"purpose": purpose}).value
#     call_app = flag_call_app_chain.invoke({"purpose": purpose}).value
#     call_prop = flag_call_prop_chain.invoke({"purpose": purpose}).value
    
#     is_cfp = bool(cfp_conf) or bool(cfp_work)
    
#     return {
#         "cfp_conf": cfp_conf,
#         "cfp_work": cfp_work,
#         "cfp_jour": cfp_jour,
#         "call_app": call_app,
#         "call_prop": call_prop,
#         "is_cfp": is_cfp
#     }


# 이 메일이 cfp 인지 분류하는 노드
def is_cfp_node(state: MailState) -> dict:
    mail_text = state.get("mail_text", "")
    is_cfp = is_cfp_chain.invoke({"mail_text": mail_text}).value
    
    return {
        "is_cfp": is_cfp
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
    return "cfp" if state.get("is_cfp", False) else "no"
    
    
    
    
graph = StateGraph(MailState)
graph.add_node("is_cfp_node", is_cfp_node)
# graph.add_node("summarize", summarize)
# graph.add_node("classify_flags", classify_flags)  # A
# graph.add_node("ext_info", ext_info)

# graph.add_edge(START, "summarize")
# graph.add_edge("summarize", "classify_flags")
# graph.add_conditional_edges(
#     "classify_flags",
#     flags_router,
#     {
#         "cfp": "ext_info",
#         "no": END
#     }
# )
# graph.add_edge("ext_info", END)
graph.add_edge(START, "is_cfp_node")
graph.add_edge("is_cfp_node", END)  

app = graph.compile()

def build_init_state(mail_text: str) -> dict:
    return {
        "mail_text": mail_text,
        "is_cfp": None,
        "purpose": None,
        "cfp_conf": None,
        "cfp_work": None,
        "cfp_jour": None,
        "call_app": None,
        "call_prop": None,
        "is_cfp": None,
        "evidence_sentences": None,
        "conf_name": None,
        "start_date": None,
        "sub_deadline": None,
        "conf_website": None
    }
    
def normalize_output(result: dict, keep_misspelled_key: bool = True) -> dict:
    return {
        "purpose": result.get("purpose"),
        "is_cfp": result.get("is_cfp"),
        "cfp_conf": result.get("cfp_conf"),
        "cfp_work": result.get("cfp_work"),
        "cfp_jour": result.get("cfp_jour"),
        "call_app": result.get("call_app"),
        "call_prop": result.get("call_prop"),
        "is_cfp": result.get("is_cfp"),
        "conference_name": result.get("conf_name"),
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





loader = TextLoader("./data/texts/95.txt", autodetect_encoding=True)
data = loader.load()


# --- 여기서 단일 파일 실행 ---
input_path = Path("./data/texts/95.txt")           # 입력 파일
output_path = Path("./single_predict.json")  # ./62_predict.json

out = process_one_file(input_path, keep_misspelled_key=True)
save_json(out, output_path)

print(f"[OK] {input_path.name} -> {output_path.name}")



# # 단일 파일에 대해 여러번 실행
# for i in range(10):
#     input_path = Path("./data/texts/95.txt")           # 입력 파일
#     output_path = Path(f"single_{i}_predict.json")

#     out = process_one_file(input_path, keep_misspelled_key=True)
#     save_json(out, output_path)

#     print(f"[OK] {input_path.name} -> {output_path.name}")