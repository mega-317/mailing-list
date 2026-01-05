from langchain_core.prompts import ChatPromptTemplate
from .common import Summary, json_parser_chain

summ_mark_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You represent an intelligent email analyzer.\n"
     "Read the email and extract key information into a JSON format.\n\n"

     "The summary doesn't matter if it's long.\n"
     "Contents related to the purpose of the mail should be included, and contents that are not related should be excluded\n"
     "The output must be in JSON format with 'summary' as the key\n\n"
     
     "Output ONLY the raw JSON object."),
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