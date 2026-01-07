# 메일 본문을 그냥 요약하도록 지시
from langchain_core.prompts import ChatPromptTemplate
from .common import json_parser_chain
import json

structural_classifier_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a JSON Structure Analyzer. Your task is to determine if a given JSON object contains a specific 'Conference Entity Structure'.\n\n"
     
     "### Target Structure Criteria:\n"
     "You are looking for a logical group (parent key) 'Conference'.\n"
     "Within this group, the following FOUR essential child keys must ALL be present (fuzzy matching allowed for names/formats):\n"
     "  1. **Name** (e.g., 'name', 'conference_name', 'event_title')\n"
     "  2. **Dates** (e.g., 'dates', 'event_period', 'schedule')\n"
     "  3. **Location or Venue** (e.g., 'location', 'venue', 'place', 'city')\n"
     "  4. **Website** (e.g., 'website', 'url', 'official_link', 'home_page')\n\n"
     
     "### Classification Logic:\n"
     "- Return `{{\"mail_type\": true}}` if a structure containing all four essential fields is found.\n"
     "- Return `{{\"mail_type\": false}}` if any of the four are missing or if the JSON does not represent a conference-like entity.\n\n"
     
     "Output ONLY the JSON object."),
    ("human",
     "### Input JSON to Analyze:\n{input_json}")
])
classifier_chain = json_parser_chain(structural_classifier_prompt)

def validation_node(state) -> dict:
    input_json = state.get("input_json", "")

    result = classifier_chain.invoke({
        "input_json": json.dumps(input_json, indent=2)
    })
    
    return {
        "mail_type": result
    }