
from langchain_core.prompts import ChatPromptTemplate
from .common import parser_chain, ConferenceUrl

ext_conf_url_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at extracting specific information from text.\n"
     "Task: From the EMAIL TEXT, extract the single, official **homepage URL** for the main conference.\n\n"
     "**Rules of Extraction:**\n"
     "1. Your goal is to find the main website for the event (e.g., `conference-name2025.org`, `event.github.io`).\n"
     "2. Just Copy the URL. Do not change at all.\n"
     "3. The output MUST be the full, complete URL. It **MUST start with 'https://' or 'http://'**.\n"
     "4. URL must end with a slash `/`.\n"
     "5. If no suitable homepage URL is found, return null for value.\n\n"
     "## CRITICAL OUTPUT FORMAT ##\n"
     "Your output MUST BE A VALID JSON OBJECT that strictly adheres to the following structure.\n"
     "The JSON object must have one key: `conf_url`.\n\n"),
    ("human", "EMAIL TEXT:\n{mail_text}\n\nReturn ONLY JSON.")
])
ext_conf_url_chain = parser_chain(ext_conf_url_prompt, ConferenceUrl)

def ext_conf_url_node(state) -> dict:
    
    infos_text = state["infos_text"]
    res: ConferenceUrl = ext_conf_url_chain.invoke({"mail_text": infos_text})
    
    conf_url_str = str(res.conf_url) if res.conf_url else None
    
    return {
        "conf_website": conf_url_str
    }