# mailflow/cfp.py
from langchain_core.prompts import ChatPromptTemplate
from .common import BoolOut, CFPLabelParser, parser_chain

cfp_candidate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert adjudicator for classifying academic emails. \n\n"
     
     "## Overall Task\n"
     "Your task is to determine if an email is a \"Call for Research Papers\" (CFP) based on mail text.\n\n"
     
     "## Input Data\n"
     "- `mail text`: The content of email.\n"
     
     "It is very likely to be CFP if the following keywords are included.\n"
     "- Specific submission deadlines (e.g., 'Papers due', 'Submission Deadline', 'Camera-Ready Deadline')\n"
     "- Direct invitations for research (e.g., 'We invite submissions of original research papers', 'Requesting a manuscript', 'Call for paper')\n"
     "- Mention of specific tracks, paper formats, or page limits (e.g., 'Research Track', 'up to 6 pages in IEEE format')\n\n"
     
     "## Output Format\n"
     "Output ONLY a strict JSON object with a single boolean key 'value'. Example: {{\"value\": true}}"),
    ("human",
     "Based on the mail text, decide if this is a Call for Papers.\n\n"
     "### Mail Text:\n{mail_text}")
])
cfp_candidate_chain = parser_chain(cfp_candidate_prompt, BoolOut)

# 무엇에 대한 cfp 메일인지 분류하는 프롬프트
classify_cfp_target_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict classifier for academic emails. Your task is to read the email and extracted text to decide which type of Call for Papers (CFP) it is.\n\n"

     "## Classes (choose exactly one):\n"
     "  - conference: a CFP for a conference\n"
     "  - workshop: a CFP for a workshop (co-located or standalone)\n"
     "  - journal: a CFP for an academic journal or special issue\n\n"
     "  - proposal: a CFP for workshop proposal (NOT for paper submission)\n"
     "  - participation: call for participation in an event\n"
     
     "Assume every email you receive here is indeed a CFP.\n"
     "Output ONLY a strict JSON object with a single key 'label' whose value is one of the classes.\n"
     "Do not include any explanations."),
    ("human",
     "Classify the following CFP based on email.\n\n"
     "### Mail text:\n{mail_text}\n\n")
])
classify_cfp_target_chain = parser_chain(classify_cfp_target_prompt, CFPLabelParser)

def cfp_candidate_node(state) -> dict:
    
    purpose = state.get("purpose")
    extracted_text = state.get("extracted_text", "")
    
    mail_text = state.get("mail_text", "")
    
    # is_cfp = cfp_candidate_chain.invoke({"purpose": purpose, "extracted_text": extracted_text}).value
    is_cfp = cfp_candidate_chain.invoke({"mail_text": mail_text}).value
    return {"cfp_candidate": is_cfp}

def classify_cfp_target_node(state) -> dict:
    purpose = state.get("purpose")
    extracted_text = state.get("extracted_text", "")
    
    mail_text = state.get("mail_text", "")
    # label = classify_cfp_target_chain.invoke({"purpose": purpose, "extracted_text": extracted_text}).label
    label = classify_cfp_target_chain.invoke({"mail_text": mail_text}).label
    
    print("CFP 분류 완료")
    
    return {
        "classify_cfp_target": label,
        "is_cfp": (label in ("conference", "workshop"))
    }
