from langchain_core.prompts import ChatPromptTemplate
from .common import BoolOut, CFPLabelParser, parser_chain

participation_validate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert adjudicator for academic emails.\n\n"
     
     "## Overall Task\n"
     "This email has already been identified as a 'Call for Participation'.\n"
     "Your specific task now is to determine **WHAT TYPE of event** the email is asking people to participate in.\n\n"
     
     "## Decision Rules:\n"
     "1. Read the `mail text` carefully. Look for the name and description of the event (e.g., 'International **Conference** on...', '**Workshop** on...', 'LATAM **School** in...').\n"
     "2. You MUST return `true` **ONLY IF** the event is clearly identified as a **Conference** or a **Workshop**.\n"
     "3. You MUST return `false` if the event is identified as a **School**, **Course**, **Summer School**, **Tutorial**, **PhD Position**, or any other type of event that is not specifically a conference or workshop.\n\n"
     
     "## Input Data\n"
     "- `mail text`: The content of the email (already known to be a Call for Participation).\n\n"
     
     "## Output Format\n"
     "Output ONLY a strict JSON object with a single boolean key 'value'. Example: {{\"value\": true}}"),
    ("human",
     "Analyze the type of event in the following Call for Participation email:\n\n"
     "### Mail Text:\n{mail_text}")
])
participation_validate_chain = parser_chain(participation_validate_prompt, BoolOut)

def participation_validate_node(state) -> dict:
    
    mail_text = state.get("mail_text", "")
    
    result = participation_validate_chain.invoke({"mail_text": mail_text}).value
    
    print("참여 요청 검증 완료")
    
    return {"is_cfp": result}