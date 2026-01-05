# 메일 본문을 그냥 요약하도록 지시
from langchain_core.prompts import ChatPromptTemplate
from .common import json_parser_chain
import json

validation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Data Quality Auditor for academic event information. Your task is to analyze a summarized JSON object "
     "and determine if it represents a valid, single-event 'Call for Papers' (CFP) based on specific criteria.\n\n"
     
     "## Validation Criteria:\n"
     "1. **is_single_main_event**: \n"
     "   - Set to `true` ONLY IF there is exactly one primary conference or event identified.\n"
     "   - Set to `false` if it is a 'Joint Conference', a list of multiple workshops, or if the main host is ambiguous.\n\n"
     
     "2. **has_event_start_date**: \n"
     "   - Set to `true` if a specific start date or period for the event is found.\n"
     "   - Set to `false` if the date is missing or only refers to deadlines.\n\n"
     
     "3. **has_submission_deadline**: \n"
     "   - Set to `true` if there is a clear deadline for paper or contribution submission.\n"
     "   - Set to `false` if no submission deadline is mentioned.\n\n"
     
     "4. **has_official_url**: \n"
     "   - Set to `true` if there is a website link or submission portal URL specifically for the event.\n"
     "   - Set to `false` if no links are provided.\n\n"
     
     "## Decision Logic:\n"
     "- **is_valid_cfp**: This is `true` ONLY IF ALL of the above four criteria are `true`.\n\n"
     
     "Output the result in the following JSON format ONLY:\n"
     "{{\n"
     "  \"is_single_main_event\": boolean,\n"
     "  \"has_event_start_date\": boolean,\n"
     "  \"has_submission_deadline\": boolean,\n"
     "  \"has_official_url\": boolean,\n"
     "  \"is_valid_cfp\": boolean,\n"
     "  \"reason\": \"A brief explanation for any 'false' values found.\"\n"
     "}}"),
    ("human",
     "### Summarized JSON to Validate:\n"
     "{summarized_json}")
])
validation_chain = json_parser_chain(validation_prompt)

def validation_node(state) -> dict:
    summary = state.get("summary", "")

    result = validation_chain.invoke({
        "summarized_json": json.dumps(summary, indent=2)
    })
    
    return {
        "validate": result
    }