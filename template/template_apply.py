# 메일 본문을 그냥 요약하도록 지시
from langchain_core.prompts import ChatPromptTemplate
from .common import json_parser_chain
from .template_from_json import create_template_from_json
import json

structure_transfer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict Data Structurer. Your task is to extract information from the email text "
     "and map it into the provided JSON template.\n\n"
     
     "## CRITICAL RULES:\n"
     "1. **Strict Structure:** You MUST use the EXACT keys and nesting structure provided in the `template_json`.\n"
     "2. **No New Keys:** Do NOT add any new keys that are not in the template.\n"
     "3. **Fill Values:** Extract relevant info from the email to fill the values.\n"
     "4. **Handle Missing Info:** If the email does not contain information for a specific key (e.g., the template has 'spanish_guidelines' but the email is only in English), set the value to `null` (or empty string/list as appropriate).\n"
     "5. **Data Types:** Respect the data types in the template (e.g., if the template has a list `[]`, output a list).\n\n"
     
     "Output ONLY the completed JSON object."),
    ("human",
     "### Template Schema (Target Structure):\n"
     "{template_json}\n\n"
     "### New Email Text (Source Content):\n"
     "{mail_text}")
])
transfer_chain = json_parser_chain(structure_transfer_prompt)

def align_structure_node(state) -> dict:
    # 1. 4.txt의 결과물(이미 처리된 JSON)을 가져옵니다. 
    # (실제 환경에서는 state에 저장되어 있거나 파일에서 로드했다고 가정)
    template = state.get("template", {})
    
    # 2. JSON에서 뼈대(Template)만 추출합니다.
    template_skeleton = create_template_from_json(template)
    
    # 3. 2.txt의 텍스트(새로운 메일)를 가져옵니다.
    target_mail_text = state.get("mail_text", "")
    
    # 4. LLM 실행 (구조 맞추기)
    aligned_json = transfer_chain.invoke({
        "template_json": json.dumps(template_skeleton, indent=2),
        "mail_text": target_mail_text
    })
    
    return {
        "aligned_summary": aligned_json
    }