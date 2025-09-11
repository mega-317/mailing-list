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
    
    # 메일 타입에 대한 검증 노드의 동의 여부
    agree_to_init_type: bool
    B_is_cfp: bool
    
    # 추가 타입 판정 노드
    mail_type_C: dict[str, bool]
    C_is_cfp: bool
    
    # 최종 판정
    a_is_cfp: bool
    b_is_cfp: bool
    c_is_cfp: bool
    final_is_cfp: bool
    vote_detail: dict[str, bool]
    
    
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





# 메일 타입을 검증하기 위한 프롬프트와 체인
inspect_cfp_prompt_B = ChatPromptTemplate.from_messages([
    ("system",
     "You verify whether the preliminary decision 'is_cfp' (True/False) is correct.\n"
     "Consider ONLY the provided EMAIL CONTENT.\n"
     "Return strict JSON with keys: agree, is_cfp.\n"
     "Guidelines:\n"
     "- 'is_cfp' should be True only if the email clearly invites submissions of conference/workshop papers "
     "(e.g., 'call for papers', 'paper submission', 'submission deadline', 'camera-ready', 'regular/short papers', 'workshop papers').\n"
     "- Journal special issues, proposals, applications/registrations without paper submission do NOT count as CfP.\n"
     "- Be conservative: if unclear, set is_cfp=false."
    ),
    ("human",
     "PRELIMINARY is_cfp: {pre_is_cfp}\n\n"
     "=== EMAIL START ===\n{mail_text}\n=== EMAIL END ===\n"
     "Return JSON with keys: agree, is_cfp.")
])
inspect_cfp_chain_B = inspect_cfp_prompt_B | llm | PydanticOutputParser(pydantic_object=CFPInspectOutput)




# 메일 타입을 검증하기 위한 프롬프트와 체인
inspect_flags_prompt_C = ChatPromptTemplate.from_messages([
    ("system",
     "You classify an academic mailing-list email into multiple labels.\n"
     "Return strict JSON that conforms to the provided schema.\n"
     "{format_instructions}\n"
     "GUIDELINES:\n"
     "- Decide using ONLY the provided sentences (excerpts from the email).\n"
     "- Set each field to true/false independently (multi-label).\n"
     "- If the sentences clearly invite paper submissions to a conference, set cfp_conf=true.\n"
     "- If they invite workshop paper submissions, set cfp_work=true.\n"
     "- If they invite journal/special issue submissions, set cfp_jour=true.\n"
     "- If they invite applications/participation/registration (not papers), set call_app=true.\n"
     "- If they invite proposals (workshop/tutorial/session/project), set call_prop=true.\n"
     "- If they mainly announce info and no direct action is requested, set info=true.\n"
     "- If unclear or outside categories, you may set etc=true.\n"
     "- Prefer precision: if uncertain, leave fields false rather than guessing."
    ),
    ("human", "SENTENCES:\n{sentences}")
])
inspect_flags_chain_C = inspect_flags_prompt_C | llm | PydanticOutputParser(pydantic_object=MailTypeFlags)




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
    return {"mail_type_A": flags}
    
    
    
def inspect_flags_B(state: MailState) -> dict:
    # A의 CfP 결론 (conference OR workshop)
    A = state.get("mail_type_A", {}) or {}
    pre_is_cfp = bool(A.get("cfp_conf", False) or A.get("cfp_work", False))

    out: CFPInspectOutput = inspect_cfp_chain_B.invoke({
        "pre_is_cfp": json.dumps(pre_is_cfp),
        "mail_text": state.get("mail_text", "")
    })
    
    print(f"기존 판정: {pre_is_cfp}")
    print(f"점검 결과: {out.is_cfp}")
    print(f"일치 여부: {pre_is_cfp == out.is_cfp}")

    # 동의 여부와 B의 CfP 판단만 state에 기록
    return {
        "agree_to_init_type": pre_is_cfp == out.is_cfp,
        "B_is_cfp": out.is_cfp,
    }



    
def inspect_flags_C(state: MailState) -> dict:
    """
    요약 노드의 evidence_sentences만 보고 멀티라벨 판정 -> CfP 여부(c_is_cfp) 계산
    """
    sents = state.get("evidence_sentences", []) or []
    if not sents:
        # 근거 문장이 없으면 보수적으로 False
        return {
            "mail_type_C": {k: False for k in LABELS},
            "C_is_cfp": False,
        }

    flags: MailTypeFlags = inspect_flags_chain_C.invoke({
        "format_instructions": PydanticOutputParser(
            pydantic_object=MailTypeFlags
        ).get_format_instructions(),
        "sentences": "\n".join(sents[:6])  # 길이 제한
    })
    flags_c = {
        "cfp_conf": flags.cfp_conf,
        "cfp_work": flags.cfp_work,
        "cfp_jour": flags.cfp_jour,
        "call_app": flags.call_app,
        "call_prop": flags.call_prop,
        "info": flags.info,
        "etc": flags.etc
    }
    c_is_cfp = bool(flags_c["cfp_conf"] or flags_c["cfp_work"])
    return {
        "mail_type_C": flags_c, 
        "C_is_cfp": c_is_cfp
    }




def majority_vote_node(state: MailState) -> dict:
    A = state.get("mail_type_A", {}) or {}
    b_is_cfp = bool(state.get("B_is_cfp", False))
    C = state.get("mail_type_C", {}) or {}

    a_is_cfp = bool(A.get("cfp_conf", False) or A.get("cfp_work", False))
    c_is_cfp = bool(state.get("C_is_cfp", False)) if C else None

    votes = [a_is_cfp, b_is_cfp]
    if c_is_cfp is not None:
        votes.append(c_is_cfp)

    true_votes = sum(1 for v in votes if v)
    false_votes = len(votes) - true_votes

    if len(votes) == 3:
        final_is_cfp = (true_votes >= 2)
    elif len(votes) == 2:
        final_is_cfp = (true_votes >= 1)  # 2자 투표일 땐 한쪽만 True여도 진행
    else:
        final_is_cfp = False

    return {
        "a_is_cfp": a_is_cfp,
        "b_is_cfp": b_is_cfp,
        "c_is_cfp": c_is_cfp,
        "final_is_cfp": final_is_cfp,
        "vote_detail": {
            "true_votes": true_votes, "false_votes": false_votes
        }
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



def need_C_or_vote(state: MailState) -> str:
    # A와 B의 CfP 결론이 다른가?
    A = state.get("mail_type_A", {}) or {}
    a_is_cfp = bool(A.get("cfp_conf", False) or A.get("cfp_work", False))
    b_is_cfp = bool(state.get("B_is_cfp", False))
    return "C" if (a_is_cfp != b_is_cfp) else "VOTE"


# 최종 라우팅: CfP면 ext_info, 아니면 END
def route_after_majority(state: MailState) -> str:
    return "ext_info" if state.get("final_is_cfp") else "END"
    
    
    
    
    
    
graph = StateGraph(MailState)
graph.add_node("summarize", summarize)
graph.add_node("classify_flags", classify_flags)  # A
graph.add_node("inspect_flags_B", inspect_flags_B)  # B
graph.add_node("inspect_flags_C", inspect_flags_C)   # C (evidence 기반)
graph.add_node("majority_vote", majority_vote_node)
graph.add_node("ext_info", ext_info)

graph.add_edge(START, "summarize")
graph.add_edge("summarize", "classify_flags")
graph.add_edge("classify_flags", "inspect_flags_B")

graph.add_conditional_edges(
    "inspect_flags_B",
    need_C_or_vote,
    {"C": "inspect_flags_C", "VOTE": "majority_vote"}
)
graph.add_edge("inspect_flags_C", "majority_vote")

graph.add_conditional_edges(
    "majority_vote",
    route_after_majority,
    {"ext_info": "ext_info", "END": END}
)

app = graph.compile()

loader = TextLoader("./data/62.txt", autodetect_encoding=True)
data = loader.load()

def build_init_state(mail_text: str) -> dict:
    return {
        "mail_text": mail_text,
        "purpose": None,
        "mail_type_A": {k: None for k in LABELS},  # ← None으로
        "agree_to_init_type": None,
        "B_is_cfp": None,
        "mail_type_C": {k: None for k in LABELS},
        "C_is_cfp": None,
        "a_is_cfp": None,
        "b_is_cfp": None,
        "c_is_cfp": None,
        "final_is_cfp": None,
        "vote_detail": None,
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
    mail_type = result.get("mail_type_A") or {k: False for k in LABELS}
    mail_type_c = result.get("mail_type_C") or {k: False for k in LABELS}
    key_conf_name = "coference_name" if keep_misspelled_key else "conference_name"

    return {
        "purpose": result.get("purpose"),
        "init_mail_type": {
            "cfp_work": mail_type.get("cfp_work"),
            "cfp_conf": mail_type.get("cfp_conf"),
            "cfp_jour": mail_type.get("cfp_jour"),
            "call_prop": mail_type.get("call_prop"),
            "call_app": mail_type.get("call_app"),
            "info":     mail_type.get("info"),
            "etc":      mail_type.get("etc"),
        },
        "agree_to_init_type": result.get("agree_to_init_type"),
        "B_is_cfp": result.get("B_is_cfp"),
        "mail_type_C": {
            "cfp_work": mail_type_c.get("cfp_work"),
            "cfp_conf": mail_type_c.get("cfp_conf", False),
            "cfp_jour": mail_type_c.get("cfp_jour", False),
            "call_prop": mail_type_c.get("call_prop", False),
            "call_app": mail_type_c.get("call_app", False),
            "info":     mail_type_c.get("info", False),
            "etc":      mail_type_c.get("etc", False),
        },
        "a_is_cfp": result.get("a_is_cfp"),
        "b_is_cfp": result.get("b_is_cfp"),
        "c_is_cfp": result.get("c_is_cfp"),
        "vote_detail": result.get("vote_detail"),
        "final_is_cfp": result.get("final_is_cfp"),
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
    state = build_init_state(data[0].page_content)
    result = app.invoke(state)

    # 결과 정규화(요청 포맷)
    out_json = normalize_output(result, keep_misspelled_key=keep_misspelled_key)
    return out_json

def save_json(obj: dict, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

# --- 여기서 단일 파일 실행 ---
# input_path = Path("./62.txt")           # 입력 파일
# output_path = input_path.with_name(f"{input_path.stem}_predict.json")  # ./62_predict.json

# out = process_one_file(input_path, keep_misspelled_key=True)
# save_json(out, output_path)

# print(f"[OK] {input_path.name} -> {output_path.name}")


# 단일 파일에 대해 여러번 실행
for i in range(20):
    input_path = Path("./data/62.txt")           # 입력 파일
    output_path = input_path.with_name(f"{input_path.stem}-{i}_predict.json")  # ./62_predict.json

    out = process_one_file(input_path, keep_misspelled_key=True)
    save_json(out, output_path)

    print(f"[OK] {input_path.name} -> {output_path.name}")