# mailflow/cfp.py
from langchain_core.prompts import ChatPromptTemplate
from .common import BoolOut, CFPLabelParser, parser_chain

# 이 메일이 CFP 메일인지 판정하는 프롬프트와 체인
cfp_candidate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an optimistic classifier. Your primary goal is to identify potential 'Call for Papers' (CFP) candidates and **aggressively avoid false negatives**. When in doubt, you MUST lean towards classifying as `true`.\n\n"
     
     "## Decision Philosophy: Asymmetric Risk\n"
     "A False Negative (missing a real CFP) is a critical failure. A False Positive (incorrectly flagging a non-CFP) is acceptable as it will be filtered later. Your logic must reflect this bias.\n\n"
     
     "## Decision Rules with Positive Bias:\n\n"
     "### **1. Decisive Positive Signal (The Override)**\n"
     "   - If the `extracted_text` contains a specific **'Submission Deadline'** for papers, the email is **always a CFP (`true`)**. \n"
     "   - This rule overrides all other negative or ambiguous signals.\n\n"

     "### **2. Strong Positive Signals**\n"
     "   - If the `purpose` explicitly states it 'invites submissions for papers/workshops/conferences', consider it a **CFP (`true`)**.\n\n"
     
     "### **3. Handling Caution Signals (Formerly Negative Signals)**\n"
     "   - Keywords like 'Call for Participation' or 'announcement' in the `purpose` are **caution signals**, not immediate rejections.\n"
     "   - **Check for contradictions**: Is the caution signal contradicted by a Decisive Positive Signal (like a deadline)? If yes, the positive signal **always wins**. \n"
     "   - Only consider classifying as `false` if these caution signals appear **AND** there is absolutely no positive evidence.\n\n"

     "### **4. Hard Exclusions (True Rejections)**\n"
     "   - Only reject if the `purpose` is clearly and exclusively about non-CFP topics like: Job Offering, Recruitment, or a Call for Proposals to ORGANIZE an event.\n\n"

     "## Output Format\n"
     "Output ONLY a strict JSON object with a single boolean key 'value'. Example: {{\"value\": true}}"),
    ("human",
     "Based on the following information and your positive-bias philosophy, decide if this is a potential Call for Papers.\n\n"
     "### Overall Purpose:\n{purpose}\n\n"
     "### Key Extracted Text:\n{extracted_text}")
])
cfp_candidate_chain = parser_chain(cfp_candidate_prompt, BoolOut)


# 무엇에 대한 cfp 메일인지 분류하는 프롬프트
classify_cfp_target_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict classifier for academic emails. Your task is to read the email's purpose and extracted text to decide which type of Call for Papers (CFP) it is.\n\n"
     
     "## Decision Strategy:\n"
     "1. **Primary Source**: Your primary source of information should be the `purpose`. It usually contains the most direct description of the event (e.g., 'This email is a call for papers for the ABC **workshop**').\n"
     "2. **Secondary Source**: Use the `extracted_text` to find additional clues or confirm your decision. For example, if the `purpose` is ambiguous, the `extracted_text` might contain 'co-located with the XYZ **Conference**' which helps classify it as a `workshop`.\n\n"

     "## Classes (choose exactly one):\n"
     "  - conference: a CFP for a conference\n"
     "  - workshop: a CFP for a workshop (co-located or standalone)\n"
     "  - journal: a CFP for an academic journal or special issue\n\n"
     
     "Assume every email you receive here is indeed a CFP.\n"
     "Output ONLY a strict JSON object with a single key 'label' whose value is one of the classes.\n"
     "Do not include any explanations."),
    ("human",
     "Classify the following CFP based on its purpose and extracted text.\n\n"
     "### Overall Purpose:\n{purpose}\n\n"
     "### Key Extracted Text:\n{extracted_text}")
])
classify_cfp_target_chain = parser_chain(classify_cfp_target_prompt, CFPLabelParser)

def cfp_candidate_node(state) -> dict:
    
    purpose = state.get("purpose")
    extracted_text = state.get("extracted_text", "")
    is_cfp = cfp_candidate_chain.invoke({"purpose": purpose, "extracted_text": extracted_text}).value
    return {"cfp_candidate": is_cfp}

def classify_cfp_target_node(state) -> dict:
    purpose = state.get("purpose")
    extracted_text = state.get("extracted_text", "")
    label = classify_cfp_target_chain.invoke({"purpose": purpose, "extracted_text": extracted_text}).label
    
    print("CFP 분류 완료")
    
    return {
        "classify_cfp_target": label,
        "is_cfp": (label in ("conference", "workshop"))
    }
