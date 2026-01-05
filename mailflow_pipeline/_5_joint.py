# mailflow/joint.py
from langchain_core.prompts import ChatPromptTemplate
from .common import BoolOut, parser_chain, JointConfResult

# is_joint_conf_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are a strict binary classifier.\n"
#      "Task: Decide if this email is about Joint Call"
#      "If the expression 'Joint Conference' appears directly, return {{\"value\": true}}.\n"
#      "If you check the expression 'Joint Call', make sure it's about the conference, and only if it's certain, return {{\"value\": true}}.\n"
#      "Otherwise, return {{\"value\": false}}.\n"
#      "Output ONLY strict JSON with key 'value'."),
#     ("human",
#      "{purpose}")
# ])

# is_joint_work_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are a strict binary classifier.\n"
#      "Task: Decide if this email is about Joint Call"
#      "If the expression 'Joint Workshop' appears directly, return {{\"value\": true}}.\n"
#      "If you check the expression 'Joint Call', make sure it's about the workshop, and only if it's certain, return {{\"value\": true}}.\n"
#      "Otherwise, return {{\"value\": false}}.\n"
#      "Output ONLY strict JSON with key 'value'."),
#     ("human",
#      "{purpose}")
# ])

is_joint_call_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier.\n"
     "Task: Decide if this email is about Joint Call"
     "If the expression 'Joint Conference' appears directly, return {{\"value\": true}}.\n"
     "If you check the expression 'Joint Call', make sure it's about the conference, and only if it's certain, return {{\"value\": true}}.\n"
     "Otherwise, return {{\"value\": false}}.\n"
     "Output ONLY strict JSON with key 'value'."),
    ("human",
     "### Mail_text: {mail_text}")
])
is_joint_call_chain = parser_chain(is_joint_call_prompt, BoolOut)
# is_joint_work_chain = parser_chain(is_joint_work_prompt, BoolOut)

is_joint_conf_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a meticulous analyst. Your task is to determine if an email is a *confusing 'Joint Conference' call* (with multiple equal hosts) OR a *focused 'Call for Workshops'* (with one single host conference).\n\n"
     
     "## Analysis Logic:\n"
     "Read the mail text (which contains a 'Joint Call' keyword) and determine the event hierarchy.\n\n"
     
     "1.  **Identify the 'Host' Conference:**\n"
     "    - First, scan the text for a single, primary **conference** (e.g., 'PROFES 2025') that is 'hosting' or 'co-locating' other events.\n\n"
     
     "2.  **Identify the 'Guest' Events:**\n"
     "    - Second, check if the email's main purpose is to list multiple **workshops** (e.g., 'QuEMaLeS 2025', 'PATH 2025') that are 'co-located with' the Host.\n\n"
     
     "3.  **Make the Final Decision:**\n"
     "    - **Case A (Host + Guests):** If you find **ONE** Host Conference and one or more Guest Workshops -> This is a focused call, **NOT** a confusing joint call.\n"
     "      - Set `is_joint_conf` to `false` and `conference_list` to `null`.\n\n"
     "    - **Case B (Multiple Hosts):** If you find **NO single Host** and the email lists multiple *main conferences* (e.g., 'TACAS', 'FASE', 'FoSSaCS') as equals -> This **IS** a confusing joint call.\n"
     "      - Set `is_joint_conf` to `true` AND extract the names of these equally listed *conferences* into `conference_list`.\n\n"
     
     "## Output Format:\n"
     "Output ONLY a strict JSON object with two keys:\n"
     "- `is_joint_conf`: A boolean (true/false).\n"
     "- `conference_list`: A list of conference names (Case B), or `null` (Case A)."),
    ("human",
     "Apply your 'Host vs. Guest' analysis logic to this mail text.\n\n"
     "### Mail_text: {mail_text}")
])
is_joint_conf_chain = parser_chain(is_joint_conf_prompt, JointConfResult)

def is_joint_call_node(state) -> dict:
    purpose = state.get("purpose", "")
    
    mail_text = state.get("mail_text", "")
    v = is_joint_call_chain.invoke({"mail_text": mail_text}).value
    
    return {
        "is_joint_call": v,
    }
    
def is_joint_conf_node(state) -> dict:
    mail_text = state.get("mail_text", "")
    
    result = is_joint_conf_chain.invoke({"mail_text": mail_text})
    is_joint_conf = result.is_joint_conf
    conf_list = result.conference_list
    
    # print(f"학회 목록: {conf_list}")
    
    is_cfp = False if is_joint_conf else True
    return {
        "is_joint_conf": is_joint_conf,
        "is_cfp": is_cfp
    }

# def is_joint_work_node(state) -> dict:
#     purpose = state.get("purpose", "")
#     v = is_joint_work_chain.invoke({"purpose": purpose}).value
#     return {"is_joint_work": v}
