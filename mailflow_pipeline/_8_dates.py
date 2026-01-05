# mailflow/names.py
from langchain_core.prompts import ChatPromptTemplate
from .common import parser_chain, ConferenceDate, SubmissionDate, NULL_STRINGS, llm, DeadlineCandidates
from langchain_core.output_parsers import StrOutputParser
import json

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
     "1. **Identify the Context/Track**: Look for the nearest heading or label that describes the group this deadline belongs to (e.g., 'Round 1', 'Round 2', 'Research Track', 'Industrial Track'). This is the most important step.\n"
     "2. **Extract the raw text**: Capture the entire relevant sentence or phrase verbatim.\n"
     "3. **Normalize the date**: Extract the date from that phrase and convert it to **YYYY-MM-DD** format.\n\n"
     
     "Look for dates associated with keywords like 'submission', 'deadline', 'due', 'track', 'paper', 'camera-ready', 'notification'.\n"
     "Exclude registration-only deadlines.\n\n"
     "If there multiple dates are mentioned in a single phrase, choose the one that extended.\n\n"
     "Your output MUST BE a JSON object with a single key 'candidates', which is a list of structured objects, each containing 'context_or_track', 'raw_text' and 'normalized_date'."),
    ("human", 
     "TEXT:\n{mail_text}\n\n"
     "Extract all potential deadlines and format them as a list of structured JSON objects.")
])
deadline_candidates_chain = parser_chain(deadline_candidates_prompt, DeadlineCandidates)

final_deadline_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly logical selection model. Your task is to analyze a JSON array of structured deadline candidates and select the single most relevant **paper submission deadline**.\n\n"
     
     "## Input Format:\n"
     "You will receive a JSON array of objects. Each object has three keys:\n"
     "- `context_or_track`: The heading or group this deadline belongs to (e.g., 'Round 1', 'Research Track').\n"
     "- `raw_text`: The original context sentence.\n"
     "- `normalized_date`: The pre-formatted date (YYYY-MM-DD).\n\n"
     
     "## Hierarchical Decision Algorithm:\n"
     "**Step 1: Filter out irrelevant candidates.**\n"
     "   - First, create a filtered list by REMOVING any candidate where the `raw_text` is about 'author notification', 'camera-ready', 'registration', or other non-submission events.\n\n"
     
     "**Step 2: Select from the filtered list using strict priority.**\n"
     "   - From the REMAINING candidates, find the best match by checking the `context_or_track` and `raw_text` fields for the following keywords **in this exact order of priority**:\n\n"
     "   - **Priority 1 (Round 1):** Look for a candidate where `context_or_track` is **'Round 1'**. If found, select its `normalized_date` and STOP.\n\n"
     "   - **Priority 2 (Research Track):** If no 'Round 1' is found, look for a candidate where `context_or_track` or `raw_text` mentions 'Research Track', 'Main Track', or 'Full Paper'.\n\n"
     "   - **Priority 3 (General Submission):** If still no match, look for a candidate where `raw_text` contains general terms like 'Paper Submission' or 'Submission Deadline'.\n\n"
     
     "**Step 3: Determine the final date.**\n"
     "   - Select the `normalized_date` of the candidate that matches the highest possible priority level.\n"
     "   - If multiple candidates match the same priority level, choose the one with the **earliest** date.\n\n"
     
     "## CRITICAL OUTPUT RULES:\n"
     "1. The final output must be a single, clean string.\n"
     "2. If a date is found, output ONLY the date in YYYY-MM-DD format.\n"
     "   - **Correct format:** 2025-05-26\n"
     "   - **Incorrect format:** \"2025-05-26\"\n" 
     "3. If no suitable submission deadline can be found, you MUST output the exact single word: **NOT_FOUND**.\n"),
    ("human",
     "## Full Context Text:\n{mail_text}\n\n"
     "## Candidate Dates (JSON Array):\n{candidate_dates}\n\n"
     "Final Answer:")
])
# 체인 구성 (단순 문자열 파서 사용)
final_deadline_chain = final_deadline_prompt | llm | StrOutputParser()

def ext_start_date_node(state) -> dict:
    
    infos_text = state["infos_text"]
    mail_text = state["mail_text"]
    res = ext_start_date_chain.invoke({"mail_text": mail_text})
    cleaned_date = res.strip()
    
    final_date = None
    if cleaned_date.lower() not in NULL_STRINGS and "not_found" not in cleaned_date.lower():
        # 날짜 형식처럼 보일 경우에만 값을 인정 (간단한 정규식 추가도 가능)
        final_date = cleaned_date

    print("학회 시작일 추출 완료")
    
    return {"start_date": final_date}


def ext_submission_deadline_node(state) -> dict:
    infos_text = state.get("infos_text", "")
    mail_text = state["mail_text"]
    conf_name = state.get("conf_name_final", "")

    # ⬇️ 체인은 이제 순수 문자열을 반환합니다.
    raw_date_str = ext_submission_deadline_chain.invoke({
        "mail_text": mail_text,
    })
    
    # ⬇️ LLM이 반환한 문자열을 정리하고, 'NOT_FOUND'나 다른 null 값인지 확인합니다.
    cleaned_date = raw_date_str.strip()
    
    final_date = None
    if cleaned_date.lower() not in NULL_STRINGS and "not_found" not in cleaned_date.lower():
        final_date = cleaned_date
    
    return {"sub_deadline": final_date}


def submission_deadline_candidates_node(state) -> dict:
    infos_text = state.get("infos_text", "")
    mail_text = state["mail_text"]
    
    result: DeadlineCandidates = deadline_candidates_chain.invoke({"mail_text": mail_text})
    
    # .model_dump()를 사용해 각 DeadlineInfo 객체를 dict로 변환합니다.
    candidates_as_dicts = [info.model_dump() for info in result.candidates]
    
    # 이제 state에는 JSON으로 변환 가능한 dict의 리스트가 저장됩니다.
    return {"sub_deadline_candidate": candidates_as_dicts}


def final_submission_deadline_node(state) -> dict:
    candidates = state.get("sub_deadline_candidate", [])
    infos_text = state.get("infos_text", "")
    mail_text = state["mail_text"]
    
    # 후보 객체 리스트에서 'normalized_date' 값만 추출하여 새로운 리스트를 만듭니다.
    dates_only_list = [c['normalized_date'] for c in candidates]
    
    # 날짜로만 구성된 리스트를 JSON 문자열로 변환합니다.
    candidate_dates_json_str = json.dumps(dates_only_list, indent=2)
    
    # 수정된 프롬프트에 JSON 문자열을 전달합니다.
    raw_date_str = final_deadline_chain.invoke({
        "candidate_dates": candidate_dates_json_str,
        "mail_text": mail_text
    })
    
    cleaned_date = raw_date_str.strip()
    
    final_date = None
    # NULL_STRINGS는 ["", "null", "none"] 등을 포함하는 리스트라고 가정합니다.
    if cleaned_date and cleaned_date.lower() not in NULL_STRINGS and "not_found" not in cleaned_date.lower():
        final_date = cleaned_date

    print("제출 마감일 추출 완료")
    
    return {"sub_deadline": final_date}