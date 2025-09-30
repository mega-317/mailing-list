# mailflow/cfp.py
from langchain_core.prompts import ChatPromptTemplate
from .common import BoolOut, CFPLabelParser, parser_chain

# 이 메일이 CFP 메일인지 판정하는 프롬프트와 체인
cfp_candidate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert adjudicator for classifying academic emails. \n\n"
     
     "## Overall Task\n"
     "Your task is to determine if an email is a \"Call for Research Papers\" (CFP) based on its summary (`purpose`) and key extracted phrases (`extracted_text`).\n\n"
     
     "## Input Data\n"
     "- `purpose`: The overall intent and context of the email.\n"
     "- `extracted_text`: Specific, verbatim phrases from the email containing submission-related keywords.\n\n"
     
     "## Decision Framework\n"
     "Follow this two-step process to make a decision:\n\n"
     
     "**Step 1: Analyze the `purpose` (Top-Down Context Check)**\n"
     "First, use the `purpose` to check for immediate deal-breakers. If the `purpose` primarily describes one of the following, it is NOT a research paper CFP. Return `false` immediately, regardless of the `extracted_text`.\n"
     "--- IMMEDIATE REJECTION IF PURPOSE IS ABOUT ---\n"
     "- School / Course / Summer School / PhD Position\n"
     "- Job Offering / Recruitment\n"
     "- Call for Participation / Registration Reminder / General Announcement\n"
     "- Call for Workshop Proposals (i.e., organizing a workshop, not submitting papers to one)\n"
     "- Call for Demos / Tutorials / Competitions\n\n"
     
     "**Step 2: Analyze the `extracted_text` (Bottom-Up Evidence Check)**\n"
     "If the email is NOT rejected in Step 1, then analyze the `extracted_text` to find conclusive evidence of a research paper CFP. \n"
     "--- STRONG CONFIRMATION IF EXTRACTED_TEXT CONTAINS ---\n"
     "- Specific submission deadlines (e.g., 'Papers due', 'Submission Deadline', 'Camera-Ready Deadline')\n"
     "- Direct invitations for research (e.g., 'We invite submissions of original research papers', 'Requesting a manuscript')\n"
     "- Mention of specific tracks, paper formats, or page limits (e.g., 'Research Track', 'up to 6 pages in IEEE format')\n\n"
     
     "## Final Verdict Logic\n"
     "- If the email is rejected in Step 1 -> `false`\n"
     "- If the email is NOT rejected in Step 1 AND contains strong confirmation signals in Step 2 -> `true`\n"
     "- Otherwise -> `false`\n\n"
     
     "## Output Format\n"
     "Output ONLY a strict JSON object with a single boolean key 'value'. Example: {{\"value\": true}}"),
    ("human",
     "Based on the following information, decide if this is a Call for Papers.\n\n"
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
