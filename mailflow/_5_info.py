# mailflow/info.py
import json
from langchain_core.prompts import ChatPromptTemplate
from .common import BoolListOut, parser_chain, split_sentences, BATCH_SIZE

info_flags_batch_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier. "
     "Given an array of sentences, return a JSON object with a key 'flags' whose value is a boolean array "
     "of the same length, where each element indicates whether the corresponding sentence contains ANY of the following:\n"
     "- A conference, symposium, or workshop name (e.g., 'ICSE 2025', 'International Conference on...').\n"
     "- An official URL (e.g., 'https://...').\n"
     "- Any key date, such as the event's start date OR various deadlines. This includes lines containing keywords like 'submission deadline', 'notification', 'registration', or specific dates like 'January 12-13, 2026'.\n"
     "- A track or session title that provides context for dates (e.g., 'Research Track', 'Industrial Track', 'Important Dates').\n\n"
     "Rules:\n"
     "- Output ONLY strict JSON with a single key 'flags'.\n"
     "- The 'flags' array MUST be the same length and order as the input list.\n"
     "- Do not include explanations or extra text."
     ),
    ("human",
     "SENTENCES(JSON array):\n{sentences_json}\n\n"
     "Return ONLY JSON like: {{\"flags\": [true, false, ...]}}")
])
info_flags_batch_chain = parser_chain(info_flags_batch_prompt, BoolListOut)

def harvest_infos_node(state) -> dict:
    mail_text = state["mail_text"]
    sentences = split_sentences(mail_text)

    picked = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i+BATCH_SIZE]
        probe = [s[:1000] for s in batch]
        sj = json.dumps(probe, ensure_ascii=False)
        try:
            flags = info_flags_batch_chain.invoke({"sentences_json": sj}).flags or []
        except Exception:
            flags = [False] * len(batch)
        for s, f in zip(batch, flags):
            if f: picked.append(s)
    return {"infos": picked}

def finalize_infos_text_node(state) -> dict:
    infos_list = state.get("infos", [])
    infos_text = "\n".join(infos_list)
    print('원문 길이: ', len(state.get("mail_text")))
    print('추출 문장 길이: ', len(infos_text))
    return {"infos_text": infos_text, "len_infos_text": len(infos_text)}
