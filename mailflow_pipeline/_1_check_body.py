from langchain_core.prompts import ChatPromptTemplate
from .common import BoolOut, CFPLabelParser, parser_chain

# 메일 본문이 제대로 있는지 확인하는 노드
check_mail_body_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict binary classifier.\n"
     "Task: Decide if the email body has meaningful content.\n"
     "Rules:\n"
     "- If the body is empty or effectively empty (only headers, boilerplate, or fewer than 3 non-empty lines), return {{\"value\": false}}.\n"
     "- Otherwise, return {{\"value\": true}}.\n"
     "Output ONLY strict JSON with key 'value'."),
    ("human", "{mail_text}")
])
check_mail_body_chain = parser_chain(check_mail_body_prompt, BoolOut)

def check_mail_body_node(state) -> dict:
    mail_text = state.get("mail_text", "")
    line_count = len(mail_text.strip().splitlines())
    
    if line_count < 100:
        result = check_mail_body_chain.invoke({"mail_text": mail_text}).value
        
        print("본문 유무 판정 완료")
        
        return {
            "has_body": result
        }