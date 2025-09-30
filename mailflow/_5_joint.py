# mailflow/joint.py
from langchain_core.prompts import ChatPromptTemplate
from .common import BoolOut, parser_chain

is_joint_conf_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier.\n"
     "Task: Decide if this email is about Joint Call"
     "If the expression 'Joint Conference' appears directly, return {{\"value\": true}}.\n"
     "If you check the expression 'Joint Call', make sure it's about the conference, and only if it's certain, return {{\"value\": true}}.\n"
     "Otherwise, return {{\"value\": false}}.\n"
     "Output ONLY strict JSON with key 'value'."),
    ("human",
     "{purpose}")
])

is_joint_work_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier.\n"
     "Task: Decide if this email is about Joint Call"
     "If the expression 'Joint Workshop' appears directly, return {{\"value\": true}}.\n"
     "If you check the expression 'Joint Call', make sure it's about the workshop, and only if it's certain, return {{\"value\": true}}.\n"
     "Otherwise, return {{\"value\": false}}.\n"
     "Output ONLY strict JSON with key 'value'."),
    ("human",
     "{purpose}")
])
is_joint_conf_chain = parser_chain(is_joint_conf_prompt, BoolOut)
is_joint_work_chain = parser_chain(is_joint_work_prompt, BoolOut)

def is_joint_conf_node(state) -> dict:
    purpose = state.get("purpose", "")
    v = is_joint_conf_chain.invoke({"purpose": purpose}).value
    return {"is_joint_conf": v}

def is_joint_work_node(state) -> dict:
    purpose = state.get("purpose", "")
    v = is_joint_work_chain.invoke({"purpose": purpose}).value
    return {"is_joint_work": v}
