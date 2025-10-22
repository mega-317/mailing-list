from langchain_core.prompts import ChatPromptTemplate
from .common import parser_chain, ConferenceUrl

ext_fallback_url_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an information extractor focused on finding fallback URLs. The primary URL extractor has failed.\n"
     "Task: From the EMAIL TEXT, find any URL that seems to provide more information or a submission link.\n\n"
     
     "**Rules of Extraction:**\n"
     "1. Look for keywords like **'More info', 'submission', 'website', 'details'** near a URL.\n"
     "2. The URL can be a shortened link (e.g., `lnkd.in`, `bit.ly`) or any other valid web address.\n"
     "3. You MUST extract the URL exactly as it appears (verbatim). Do not modify it.\n"
     "4. The output MUST be a full URL starting with 'https://' or 'http://'.\n\n"
     
     "## CRITICAL OUTPUT FORMAT ##\n"
     "Your output MUST BE A VALID JSON OBJECT that strictly adheres to the following structure.\n"
     "The JSON object must have one key: `conf_url`.\n\n"),
    ("human", "EMAIL TEXT:\n{mail_text}\n\nReturn ONLY JSON.")
])
ext_fallback_url_chain = parser_chain(ext_fallback_url_prompt, ConferenceUrl)

def ext_fallback_url_node(state) -> dict:
    
    infos_text = state["infos_text"]
    mail_text = state["mail_text"]
    res: ConferenceUrl = ext_fallback_url_chain.invoke({"mail_text": mail_text})
    
    conf_url_str = str(res.conf_url) if res.conf_url else None
    
    return {
        "conf_website": conf_url_str
    }