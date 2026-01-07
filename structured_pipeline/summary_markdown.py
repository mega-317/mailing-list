from langchain_core.prompts import ChatPromptTemplate
from .common import Summary, json_parser_chain

summ_mark_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You represent an expert Information Extraction System.\n"
     "Your goal is to parse the email text and transform it into a highly structured JSON format.\n\n"

     "### Extraction Strategy:\n"
     "- **Hierarchical Cohesion:** Group all related attributes directly under their corresponding entity.\n"

     "### Output Rules:\n"
     "- **Strict JSON Syntax:** Output ONLY the raw JSON object. Do not include any comments (//), markdown code blocks (```json), or conversational filler.\n\n"
     ),
    ("human",
     "{mail_text}"),
])
summ_mark_chain = json_parser_chain(summ_mark_prompt)

def summ_mark_node(state) -> dict:

    mail_text = state.get("mail_text", "").strip()
    result_dict = summ_mark_chain.invoke({"mail_text": mail_text})

    return {
        "summary_dict": result_dict
    }