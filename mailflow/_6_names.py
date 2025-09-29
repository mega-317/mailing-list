# mailflow/names.py
from langchain_core.prompts import ChatPromptTemplate
from .common import ExtractName, ConfChoice, parser_chain

ext_conf_name_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an information extraction model.\n"
     "Task: From the EMAIL TEXT, extract ALL potential names and acronyms for conferences or symposiums. "
     "Your goal is to be comprehensive and extract all plausible candidates for later filtering.\n\n"
     "Rules:\n"
     "1. Extract both long-form names (e.g., 'International Conference on Software Engineering') and short-form acronyms with years (e.g., 'ICSE 2025').\n"
     "2. **CRITICAL PARSING RULE**: When filling the JSON, the 'acronym' field must contain ONLY the alphabetic abbreviation (e.g., 'FROM', 'ICSE'). The 'year' field must contain the corresponding number (e.g., 2025). **NEVER include the year inside the 'acronym' field.**\n"
     "3. If a single piece of text contains both an acronym and a full name (e.g., 'FROM 2025 - 9th Working Formal Methods Symposium'), extract the entire text as 'raw', but correctly parse the 'acronym' ('FROM') and 'year' (2025) fields from it.\n"
     "4. Do not include keywords like IEEE in the ACRONYM.\n"
     "5. Do not include numbers in the ACRONYM.\n"
     "6. evidence MUST be copied verbatim from near the mention.\n"
     "7. Return STRICT JSON for the schema:\n{schema}"),
    ("human", "EMAIL:\n{mail_text}\n\nReturn ONLY JSON.")
]).partial(schema=ExtractName.model_json_schema())
ext_conf_name_chain = parser_chain(ext_conf_name_prompt, ExtractName)

ext_work_name_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an information extraction model.\n"
     "Task: From the given EMAIL TEXT, extract ONLY the names of WORKSHOPS or SUB-EVENTS (co-located tracks under a larger event). "
     "EXCLUDE the host/main conference or umbrella event (e.g., JOWO, FOIS) and EXCLUDE journals/publishers/venues like PACM, TOPLAS, ACM DL.\n\n"
     "Also EXCLUDE conference name or names related to them.\n"
     "Normalization rules:\n"
     "- Prefer '<ACRONYM> <YEAR>' if present (e.g., 'IFOW 2025'). If year is missing, return just the acronym (e.g., 'FOMI', 'PLATO').\n"
     "- Acronym MAY be UPPERCASE, CamelCase, or mixed with digits (e.g., 'PwM2', 'Shields 2').\n"
     "- Year MAY be a 4-digit number (2025) or 2-digit number (e.g., 25) or a Roman numeral/ordinal (e.g., 'IX', '2nd').\n"
     "- evidence MUST be copied verbatim from near the mention.\n"
     "- Ignore email headers/boilerplate like 'Subject:', 'From:', 'Date:', plain numbering lines like '2.' or '3.'.\n\n"
     "Return STRICT JSON for the schema:\n{schema}"),
    ("human", "EMAIL TEXT:\n{mail_text}\n\nReturn ONLY JSON.")
]).partial(schema=ExtractName.model_json_schema())
ext_work_name_chain = parser_chain(ext_work_name_prompt, ExtractName)

final_conf_name_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict selector.\n"
     "Task: Given an EMAIL TEXT and a list of candidate event titles, choose the single event that is the **primary subject** of the email's Call for Papers (CfP).\n\n"
     "**Decision Hierarchy (Follow in this order of importance):**\n"
     "1.  **Subject Line is King:** The event mentioned prominently in the email's 'Subject:' line is the highest priority. This is the most reliable indicator of the email's main topic.\n"
     "2.  **Workshops Can Be the Main Subject:** The primary subject can be a conference, symposium, OR a workshop. If the email's main purpose is a CfP for a workshop (e.g., the title is 'Call for Papers for Workshop X'), then that workshop **IS** the correct choice.\n"
     "3.  **Interpret 'Co-location' as Context:** If Candidate A is described as 'co-located with' or 'part of' Candidate B, this means Candidate A (the workshop/sub-event) is the primary subject, and Candidate B is just providing context about the location/venue. **You must choose Candidate A.** Do NOT choose the contextual (umbrella) venue.\n\n"
     "**Output Rules:**\n"
     "- The answer MUST be **exactly one** string copied **verbatim** from the `raw` field of one of the candidates.\n"
     "- Output ONLY strict JSON: {{\"choice\": <one-of-candidates>}} with no extra text."
     ),
    ("human",
     "EMAIL TEXT:\n{mail_text}\n\n"
     "CANDIDATES:\n{candidates}\n"
     "Return ONLY JSON.")
])
final_conf_name_chain = parser_chain(final_conf_name_prompt, ConfChoice)

def ext_conf_name_node(state) -> dict:
    
    infos_text = state["infos_text"]
    res: ExtractName = ext_conf_name_chain.invoke({"mail_text": infos_text})
    cands = [c.model_dump() for c in res.Name_candidates]
    return {"conf_name_candidates": cands}

def ext_work_name_node(state) -> dict:
    
    infos_text = state["infos_text"]
    res: ExtractName = ext_work_name_chain.invoke({"mail_text": infos_text})
    cands = [c.model_dump() for c in res.Name_candidates]
    return {"work_name_candidates": cands}

def build_conf_tokens_node(state) -> dict:
    tokens_conf, tokens_work = set(), set()
    for c in state.get("conf_name_candidates", []):
        acr, yr = c.get("acronym"), c.get("year")
        if acr and yr: tokens_conf.add(f"{acr} {int(yr)}")
    for c in state.get("work_name_candidates", []):
        acr, yr = c.get("acronym"), c.get("year")
        if acr and yr: tokens_work.add(f"{acr} {int(yr)}")
        elif acr:      tokens_work.add(f"{acr}")
    return {"conf_tokens": sorted(tokens_conf), "work_tokens": sorted(tokens_work)}

def final_conf_name_node(state) -> dict:
    
    info_text = state.get("infos_text")
    candidates = state.get("conf_tokens")
    choice = final_conf_name_chain.invoke({"mail_text": info_text, "candidates": candidates}).choice
    return {"conf_name_final": choice}
