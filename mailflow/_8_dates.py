# mailflow/names.py
from langchain_core.prompts import ChatPromptTemplate
from .common import parser_chain, ConferenceDate, SubmissionDate, NULL_STRINGS, llm, DeadlineCandidates
from langchain_core.output_parsers import StrOutputParser
import json

# ext_start_date_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are an expert at extracting specific information from text.\n"
#      "Task: From the EMAIL TEXT, extract the main conference's start date.\n\n"
#      "Rules:\n"
#      "1.  The date MUST be normalized to the **YYYY-MM-DD** format.\n"
#      "2.  You MUST extract the date the conference **begins**. Look for phrases like **'take place on', 'held from', 'during', 'dates are'** to identify the event period.\n"
#      "3.  If the conference runs for multiple days (e.g., 'January 12-14, 2026'), extract only the first day (2026-01-12).\n"
#      "4.  If no specific start date for the main conference is found, return null.\n\n"
#      "Return STRICT JSON for the schema:\n{schema}"),
#     ("human", "EMAIL TEXT:\n{mail_text}\n\nReturn ONLY JSON.")
# ]).partial(schema=ConferenceDate.model_json_schema())
# ext_start_date_chain = parser_chain(ext_start_date_prompt, ConferenceDate)
ext_start_date_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly accurate information extraction model.\n"
     "Task: From the provided TEXT, find the specific start date for the conference.\n\n"
     "**CRITICAL OUTPUT RULES:**\n"
     "1. Your output MUST BE ONLY the date string, normalized to the **YYYY-MM-DD** format. (e.g., 2026-02-16)\n"
     "2. **DO NOT** output JSON, explanations, or any text other than the date itself.\n"
     "3. If a start date for conference cannot be found for any reason, you MUST output the exact single word: **NOT_FOUND**\n\n"
     "**Extraction Logic:**\n"
     "- Focus ONLY on the date the conference **begins**.\n"
     "- Look for phrases like 'take place on', 'held from', 'during', or the date appearing directly next to the conference name.\n"
     "- If the event spans multiple days (e.g., 'February 16-18, 2026'), you must extract only the very first day (2026-02-16).\n"
     ),
    ("human", "TEXT:\n{mail_text}\n\nReturn ONLY the date string in YYYY-MM-DD format or the word NOT_FOUND.")
])
ext_start_date_chain = ext_start_date_prompt | llm | StrOutputParser()

ext_submission_deadline_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly accurate information extraction model.\n"
     "Task: From the provided TEXT, find the paper **submission deadline** for the conference.\n\n"
     "**CRITICAL OUTPUT RULES:**\n"
     "1. Your output MUST BE ONLY the date string, normalized to the **YYYY-MM-DD** format. (e.g., 2025-09-28)\n"
     "2. **DO NOT** output JSON, explanations, or any text other than the date itself.\n"
     "3. If a submission deadline for conference cannot be found, you MUST output the exact single word: **NOT_FOUND**\n\n"
     "**Extraction Logic:**\n"
     "- Look for keywords like 'Paper Submission', 'Submission deadline', 'papers due'.\n"
     "- The text might have multiple deadlines (e.g., 'abstract', 'camera-ready'). Prioritize the main 'full paper' or general submission deadline.\n"
     "- IGNORE other dates like 'notification date' or 'registration deadline'.\n"
     ),
    ("human", "TEXT:\n{mail_text}\n\nReturn ONLY the date string in YYYY-MM-DD format or the word NOT_FOUND.")
])
ext_submission_deadline_chain = ext_submission_deadline_prompt | llm | StrOutputParser()

deadline_candidates_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly accurate information extraction model. Your task is to find all potential paper submission deadlines from the given text and structure them.\n\n"
     "For each deadline phrase you find, you must perform two actions:\n"
     "1. **Extract the raw text**: Capture the entire relevant sentence or phrase verbatim.\n"
     "2. **Normalize the date**: Extract the date from that phrase and convert it to **YYYY-MM-DD** format.\n\n"
     "Look for dates associated with keywords like 'submission', 'deadline', 'due', 'track', 'paper', 'camera-ready', 'notification'.\n"
     "Exclude registration-only deadlines.\n\n"
     "Your output MUST BE a JSON object with a single key 'candidates', which is a list of structured objects, each containing 'raw_text' and 'normalized_date'."),
    ("human", 
     "TEXT:\n{mail_text}\n\n"
     "Extract all potential deadlines and format them as a list of structured JSON objects.")
])
deadline_candidates_chain = parser_chain(deadline_candidates_prompt, DeadlineCandidates)

final_deadline_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a meticulous researcher. Your task is to identify the single most important **Research Paper Submission Deadline** by analyzing a full text and then selecting the correct date from a provided list of candidate dates.\n\n"
     
     "## Input Data:\n"
     "- `full_context_text`: The original email text containing all contextual clues and section headings.\n"
     "- `candidate_dates_json`: A simple JSON array of valid, pre-normalized dates (e.g., `[\"2025-10-16\", \"2025-11-17\", ...]`) that you can choose from.\n\n"
     
     "## Step-by-Step Analysis Logic:\n"
     "1.  **Identify the Most Important Section**: Your highest priority is the section labeled **'Research Track'**, 'Main Track', or 'Full Papers'. Scan the `full_context_text` to find this section.\n"
     "2.  **Find the Deadline within that Section**: Inside that priority section, locate the phrase 'Paper Submission Deadline' or a similar term.\n"
     "3.  **Match the Date**: Find the date mentioned on that specific line (e.g., '16 October, 2025') and find its exact `YYYY-MM-DD` equivalent in the provided `candidate_dates_json` list. This is your answer.\n"
     "4.  **Fallback (If No Research Track)**: If no 'Research Track' is found, search the entire `full_context_text` for the most prominent general 'Paper Submission Deadline' and match its date from the `candidate_dates_json` list.\n"
     "5.  **Final Check**: Ensure the date you choose is for a paper submission, not a notification, camera-ready, or proposal deadline.\n\n"
     
     "## CRITICAL OUTPUT RULES:\n"
     "1. Your output MUST BE ONLY the selected date string from the provided list (e.g., \"2025-10-16\").\n"
     "2. After following the analysis logic, your final output MUST BE ONLY the resulting date string in YYYY-MM-DD format (e.g., \"2025-10-16\") OR the single word NOT_FOUND.\n"
     "3. DO NOT include your reasoning, explanations, steps, or any other text in the final output.** Only provide the single, final answer."),
    ("human",
     "## Full Context Text:\n{infos_text}\n\n"
     "## Candidate Dates (JSON Array):\n{candidate_dates}")
])
# 체인 구성 (단순 문자열 파서 사용)
final_deadline_chain = final_deadline_prompt | llm | StrOutputParser()

def ext_start_date_node(state) -> dict:
    
    infos_text = state["infos_text"]
    res = ext_start_date_chain.invoke({"mail_text": infos_text})
    cleaned_date = res.strip()
    
    final_date = None
    if cleaned_date.lower() not in NULL_STRINGS and "not_found" not in cleaned_date.lower():
        # 날짜 형식처럼 보일 경우에만 값을 인정 (간단한 정규식 추가도 가능)
        final_date = cleaned_date
    
    print(f"추출된 시작일 RAW 문자열: '{res}'")
    print(f" -> 최종 저장될 시작일: {final_date}")
    
    return {"start_date": final_date}


def ext_submission_deadline_node(state) -> dict:
    infos_text = state.get("infos_text", "")
    conf_name = state.get("conf_name_final", "")
    
    if not infos_text or not conf_name:
        return {"sub_deadline": None}

    # ⬇️ 체인은 이제 순수 문자열을 반환합니다.
    raw_date_str = ext_submission_deadline_chain.invoke({
        "mail_text": infos_text,
    })
    
    # ⬇️ LLM이 반환한 문자열을 정리하고, 'NOT_FOUND'나 다른 null 값인지 확인합니다.
    cleaned_date = raw_date_str.strip()
    
    final_date = None
    if cleaned_date.lower() not in NULL_STRINGS and "not_found" not in cleaned_date.lower():
        final_date = cleaned_date

    print(f"추출된 제출 마감일 RAW 문자열 ({conf_name}): '{raw_date_str}'")
    print(f" -> 최종 저장될 제출 마감일: {final_date}")
    
    return {"sub_deadline": final_date}


def submission_deadline_candidates_node(state) -> dict:
    infos_text = state.get("infos_text", "")
    if not infos_text:
        return {"sub_deadline_candidate": []}
    
    result: DeadlineCandidates = deadline_candidates_chain.invoke({"mail_text": infos_text})
    
    # .model_dump()를 사용해 각 DeadlineInfo 객체를 dict로 변환합니다.
    candidates_as_dicts = [info.model_dump() for info in result.candidates]
    
    # 이제 state에는 JSON으로 변환 가능한 dict의 리스트가 저장됩니다.
    return {"sub_deadline_candidate": candidates_as_dicts}


def final_submission_deadline_node(state) -> dict:
    candidates = state.get("sub_deadline_candidate", [])
    infos_text = state.get("infos_text", "")
    if not candidates:
        return {"sub_deadline": None}
    
    # 후보 객체 리스트에서 'normalized_date' 값만 추출하여 새로운 리스트를 만듭니다.
    dates_only_list = [c['normalized_date'] for c in candidates]
    
    # 날짜로만 구성된 리스트를 JSON 문자열로 변환합니다.
    candidate_dates_json_str = json.dumps(dates_only_list, indent=2)
    
    # 수정된 프롬프트에 JSON 문자열을 전달합니다.
    raw_date_str = final_deadline_chain.invoke({
        "candidate_dates": candidate_dates_json_str,
        "infos_text": infos_text
    })
    
    cleaned_date = raw_date_str.strip()
    
    final_date = None
    # NULL_STRINGS는 ["", "null", "none"] 등을 포함하는 리스트라고 가정합니다.
    if cleaned_date and cleaned_date.lower() not in NULL_STRINGS and "not_found" not in cleaned_date.lower():
        final_date = cleaned_date

    print(f"최종 선택된 제출 마감일 RAW 문자열: '{raw_date_str}'")
    print(f" -> 최종 저장될 제출 마감일: {final_date}")
    
    return {"sub_deadline": final_date}