# mailflow/names.py
from langchain_core.prompts import ChatPromptTemplate
from .common import ExtractName, ConfChoice, parser_chain

ext_conf_name_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an information extraction model.\n"
     "Task: From the email, Extract the name of the Conference or Host Event. Ignore the name of the sub-event or workshop.\n"
     "STRICTLY EXCLUDE journals, journal series, publishers, and venues like PACM, TOPLAS, HCI journal names, ACM DL pages, etc.\n"
     "Rules:\n"
     "- Prefer '<ACRONYM> <YEAR>' if present (e.g., 'EICS 2026').\n"
     "- If a long-form event name appears on the same line (or neighboring line) with an acronym and a year.\n"
     "- Do not include keywords like IEEE in ACRONYM."
     "- evidence MUST be copied verbatim from near the mention.\n"
     "- Return STRICT JSON for the schema:\n{schema}"),
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
     "Task: Given EMAIL TEXT and a JSON array of candidate conference titles, "
     "choose the single main/host conference mentioned in the email.\n"
     "Rules:\n"
     "- Choose the **host/main conference** (not a workshop, not a sub-event/track).\n"
     "- Especially, choose the one that is mentioned in the title.\n"
     "- Prefer the umbrella venue when candidates include an event and its sub-events.\n"
     "- The answer MUST be **exactly one** string copied **verbatim** from the candidates list.\n"
     "- If none is actually mentioned as the host/main venue, pick the one most clearly presented as the main venue in the email.\n"
     "- Output ONLY strict JSON: {{\"choice\": <one-of-candidates>}} with no extra text."
    ),
    ("human",
     "EMAIL TEXT:\n{mail_text}\n\n"
     "CANDIDATES:\n{candidates}\n"
     "Return ONLY JSON.")
])
final_conf_name_chain = parser_chain(final_conf_name_prompt, ConfChoice)

def ext_conf_name_node(state) -> dict:
    print("학회 이름 추출")
    
    infos_text = state["infos_text"]
    res: ExtractName = ext_conf_name_chain.invoke({"mail_text": infos_text})
    cands = [c.model_dump() for c in res.Name_candidates]
    return {"conf_name_candidates": cands}

def ext_work_name_node(state) -> dict:
    print("워크숍 이름 추출")
    
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
    print("학회 이름 결정")
    info_text = state.get("infos_text")
    candidates = state.get("conf_tokens")
    choice = final_conf_name_chain.invoke({"mail_text": info_text, "candidates": candidates}).choice
    return {"conf_name_final": choice}
