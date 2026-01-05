# 메일 본문을 그냥 요약하도록 지시
from langchain_core.prompts import ChatPromptTemplate
from .common import str_parser_chain

summ_str_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict content filter. Your task is to reconstruct the email body by extracting ONLY the sentences directly related to the email's Subject and Purpose.\n\n"
     
     "## Instructions:\n"
     "1. **Analyze the Subject:** Identify the core topic of the email (e.g., a specific conference CFP, a workshop invitation).\n"
     "2. **Extract Relevant Sentences:** Select sentences from the body that provide factual details about that topic (Event names, Dates, Submission instructions, Topic descriptions, URLs).\n"
     "3. **Filter Noise:** Remove greetings ('Dear...'), closings ('Best regards...'), signatures, and irrelevant marketing fluff.\n"
     "4. **Preserve Flow:** Stitch the extracted sentences together into **natural language paragraphs**.\n\n"
     
     "## Constraints:\n"
     "- **DO NOT use JSON.**\n"
     "- **DO NOT use bullet points or markdown headers.**\n"
     "- **DO NOT rewrite or summarize.** Keep the original phrasing of the sentences to preserve context.\n"
     "- Output ONLY the reconstructed plain text."),
    ("human",
     "{mail_text}"),
])
summ_str_chain = str_parser_chain(summ_str_prompt)

def summ_str_node(state) -> dict:

    mail_text = state.get("mail_text", "").strip()
    result_text = summ_str_chain.invoke({"mail_text": mail_text})

    return {
        "summary": result_text
    }