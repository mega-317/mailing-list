# mailflow/names.py
from langchain_core.prompts import ChatPromptTemplate
from .common import parser_chain, ConferenceDate, SubmissionDate, NULL_STRINGS, llm
from langchain_core.output_parsers import StrOutputParser

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

# ⬇️ Pydantic 파서 대신, LLM의 응답을 그대로 문자열로 받는 체인을 정의합니다.
ext_submission_deadline_chain = ext_submission_deadline_prompt | llm | StrOutputParser()


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