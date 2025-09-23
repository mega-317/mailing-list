# mailflow/cfp.py
from langchain_core.prompts import ChatPromptTemplate
from .common import BoolOut, CFPLabelParser, parser_chain

# 이 메일이 CFP 메일인지 판정하는 프롬프트와 체인
cfp_candidate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier for academic emails.\n"
     "Task: I'll give you mail's purpose. Determine if the email is a Call for Papers (CFP).\n"
     "Definition of CFP:\n"
     "- Asking to submit a paper for a conference or workshop\n"
     "- Requesting a manuscript for a journal special issue\n\n"
     "- Referring to 'Track', 'Submission', 'Paper'."
     "Proposal mail is not applicable, such as requesting a proposal for a workshop or satellite event.\n"
     "If yes → return JSON {{\"value\": true}}\n"
     "If no → return JSON {{\"value\": false}}\n"
     "Rules:\n"
     "- Output ONLY strict JSON with key 'value'.\n"
     "- Never add explanations or extra text."),
    ("human",
     "Classify the following email based on mail's purpose:\n"
     "{purpose}")
])
cfp_candidate_chain = parser_chain(cfp_candidate_prompt, BoolOut)


# 무엇에 대한 cfp 메일인지 분류하는 프롬프트
classify_cfp_purpose_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict classifier for academic emails.\n"
     "Task: Read the purpose of email and decide which type of Call for Papers (CFP) it is.\n"
     "Classes (choose exactly one):\n"
     "  - conference  : a CFP for a conference\n"
     "  - workshop    : a CFP for a workshop (co-located or standalone)\n"
     "  - journal     : a CFP for an academic journal or special issue (edited books/book chapters also belong here)\n\n"
     "Assume every email you receive here is indeed a CFP (no need to reject).\n"
     "Output ONLY a strict JSON object with a single key 'label' whose value is one of: "
     "'conference', 'workshop', 'journal'.\n"
     "Do not include any explanations or extra text."
    ),
    ("human",
     "Classify the following email based on mail's purpose:\n"
     "{purpose}")
])
classify_cfp_purpose_chain = parser_chain(classify_cfp_purpose_prompt, CFPLabelParser)

def cfp_candidate_node(state) -> dict:
    print("cfp 분류 시작")
    
    purpose = state.get("purpose", "")
    is_cfp = cfp_candidate_chain.invoke({"purpose": purpose}).value
    return {"cfp_candidate": is_cfp}

def classify_cfp_purpose_node(state) -> dict:
    purpose = state.get("purpose", "")
    label = classify_cfp_purpose_chain.invoke({"purpose": purpose}).label
    return {
        "classify_cfp_purpose": label,
        "is_cfp_purpose": (label in ("conference", "workshop"))
    }
