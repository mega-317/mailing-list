# 메일 본문을 그냥 요약하도록 지시
from langchain_core.prompts import ChatPromptTemplate
from .common import json_parser_chain
import json

entity_restructuring_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Data Architect specialized in Information Extraction. Your task is to reorganize the raw email text into a structured format centered around 'Entities' (the primary purposes or opportunities mentioned).\n\n"
     
     "### 1. Goal:\n"
     "Identify every distinct 'Purpose' or 'Event' in the email (e.g., a Conference, a Job Opening, a Workshop, a Call for Papers). For each entity found, group all related metadata (dates, requirements, links) directly under that entity to ensure maximum cohesion.\n\n"
     
     "### 2. Restructuring Rules:\n"
     "- **Identify Entities First:** Determine the main subject(s). If it's a 'Job Proposal', make that the root. If it's a 'Conference', make that the root.\n"
     "- **Functional Grouping:** Place all specific details (Submission dates, Job requirements, Eligibility) as sub-attributes of the relevant entity.\n"
     "- **Support Multiple Entities:** If an email mentions both a Main Conference and a Job Opening, create two separate root objects. This is critical for detecting 'Joint' or 'Newsletter' type outliers.\n"
     "- **Contextual Preservation:** Do not just list keywords. Maintain the relationship between data points (e.g., link a specific salary to a specific job position).\n\n"
     
     "### 3. Strict JSON Rules (Prevent Parsing Errors):\n"
     "- **No Comments:** Never include comments like `//` or `/* */` inside the JSON output.\n"
     "- **Valid Syntax:** Ensure all strings are enclosed in double quotes and trailing commas are avoided.\n"
     "- **No Explanations:** Output ONLY the raw JSON code block, without any conversational text before or after.\n\n"

     "### 4. Output Format:\n"
     "Return a JSON object with an `extracted_entities` list. Each entity must have a `type`, `name`, and its specific `details` object."),
    ("human",
     "### Raw Email Text:\n{mail_text}")
])
entity_chain = json_parser_chain(entity_restructuring_prompt)

def entity_node(state) -> dict:
    mail_text = state.get("mail_text", "")

    result = entity_chain.invoke({
        "mail_text": mail_text
    })
    
    return {
        "result": result
    }