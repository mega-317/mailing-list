# 메일 본문을 그냥 요약하도록 지시
from langchain_core.prompts import ChatPromptTemplate
from .common import json_parser_chain
import json

validation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Data Quality Auditor for academic events. Your goal is to identify if the JSON represents a **Single Actionable Target**.\n\n"
     
     "## 1. Validation Criteria (Functional Dominance):\n\n"
     
     "### A. is_hierarchically_clear\n"
     "- `true`: There is one dominant event. Even if other events are mentioned (e.g., co-located, hosted by), they serve as 'contextual background' without competing for the primary focus.\n"
     "- **Critical Rule:** An event is NOT a 'Peer' if it lacks actionable data (Deadlines, Dates, URL). A mention of a co-located event (e.g., POPL 2026) without its own deadlines is NOT a peer relationship; it is a nested or co-located context.\n"
     "- `false`: Two or more events are listed with equal weight, and it is unclear which one is the primary subject.\n\n"
     
     "### B. is_functionally_singular\n"
     "- `true`: Exactly ONE event possesses the full set of 'Actionable Data' (Specific Submission Deadlines + Start/End Dates).\n"
     "- `false`: Multiple events each provide their own specific submission deadlines and schedules.\n\n"
     
     "### C. has_essential_info\n"
     "- `true`: The primary target event has a Start Date, Submission Deadline, and Official URL.\n"
     "- `false`: Any of these three are missing for the main target.\n\n"
     
     "## 2. Decision Logic:\n"
     "- **is_valid_cfp**: `true` if ALL criteria (A, B, C) are `true`.\n"
     "- If an event (like VMCAI 2026) has all data, and its co-located partner (like POPL 2026) has NO data, `is_hierarchically_clear` MUST be `true`.\n\n"
     
     "## 3. Output Format:\n"
     "Output ONLY the following JSON:\n"
     "{{\n"
     "  \"is_hierarchically_clear\": boolean,\n"
     "  \"is_functionally_singular\": boolean,\n"
     "  \"has_essential_info\": boolean,\n"
     "  \"is_valid_cfp\": boolean,\n"
     "  \"detected_pattern\": \"Single / Nested / Co-located / Peer-Multi\",\n"
     "  \"reason\": \"Explain why the relationship is clear or ambiguous based on the distribution of actionable data.\"\n"
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