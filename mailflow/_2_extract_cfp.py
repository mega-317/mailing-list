# mailflow/extract_cfp.py
from langchain_core.prompts import ChatPromptTemplate
from .common import ExtractText, parser_chain

extract_text_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert information extractor. Your task is to find and extract sentences or short paragraphs from the given email text that are directly related to a Call for Papers. "
     "You must extract two types of information:\n"  # <-- 1. 추출 대상을 두 종류로 명확히 나눔
     "1. **Event Identity**: Sentences containing the official name of the conference, workshop, or journal, its main theme, and its date/location (e.g., 'MLOps25, Workshop on Machine Learning Operations, co-located with ECAI 2025...').\n"
     "2. **Submission Details**: Sentences containing keywords like 'submission', 'deadline', 'due date', 'call for papers', 'manuscript', 'author instructions', 'important dates'.\n\n"
     
     "Return the extracted text snippets verbatim as a single string, with each snippet separated by ','. "
     "Your output MUST BE a JSON object with a single key 'extracted_text' whose value is the combined string."),
    ("human",
     "From the following email, extract all key sentences about the event's identity and paper submission details:\n"  # <-- 2. human 프롬프트도 약간 더 구체적으로 변경
     "=== EMAIL START ===\n{mail_text}\n=== EMAIL END ===")
])
extract_text_chain = parser_chain(extract_text_prompt, ExtractText)

def extract_text_node(state) -> dict:
    
    mail_text = state.get("mail_text", "").strip()
    out: ExtractText = extract_text_chain.invoke({"mail_text": mail_text})
    return {
        "extracted_text": out.extracted_text,
        "len_mail_text": len(mail_text),
        "len_extracted_text": len(out.extracted_text),
    }
