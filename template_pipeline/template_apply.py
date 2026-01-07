# 메일 본문을 그냥 요약하도록 지시
from langchain_core.prompts import ChatPromptTemplate
from .common import json_parser_chain
from .template_from_json import create_template_from_json
import json

# 주어진 템플릿에 맞게 메일을 요약하는 프롬프트
structure_transfer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert Knowledge Graph Engineer. Your task is to extract information from the email text "
     "and map it into the provided JSON template while **preserving the inherent complexity and independence of entities**.\n\n"
     
     "## CRITICAL DATA PRESERVATION RULES:\n"
     "1. **Anchor Consistency:** Use the top-level keys provided in the `template_json` as fixed categories (Anchors).\n"
     "2. **Entity Independence (Joint Conf Support):** If the email contains multiple independent entities (e.g., co-located conferences, multiple workshops, or distinct tracks), **DO NOT merge them**. Represent them as a **List of Objects** within the relevant key to preserve their individual identities.\n"
     "3. **Structural Autonomy:** Within each top-level key, you have the freedom to create nested JSON structures, lists, or strings that best represent the original data's hierarchy. Do not flatten data if it results in information loss.\n"
     "4. **No Lossy Compression:** Do not 'summarize away' the specific relationships (e.g., linking a specific deadline to a specific track). Maintain the mapping using nested objects.\n"
     "5. **Missing Info:** If a category has no relevant data, set it to `null`. Do not invent or assume values.\n\n"
     
     "Output ONLY the completed JSON object."),
    ("human",
     "### Template Schema (Top-level Anchors):\n"
     "{template_json}\n\n"
     "### New Email Text (Source Content):\n"
     "{mail_text}")
])
transfer_chain = json_parser_chain(structure_transfer_prompt)


# 큰 틀의 템플릿은 유지하고 하위 구조는 동적으로 생성/배치하여 요약하는 프롬프트
flexible_summary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Knowledge Graph Engineer. Your task is to extract information from an email and map it into a provided JSON template, while maintaining the logical hierarchy of the data.\n\n"
     
     "### CRITICAL RULES:\n"
     "1. **Anchor Preservation:** You MUST use the top-level keys provided in the `template_schema` as your primary categories.\n"
     "2. **Structural Autonomy:** Inside each top-level key, you have full freedom to design the data structure. Use nested JSON objects, lists, or strings to best represent the complexity of the information.\n"
     "3. **Entity Integrity (Joint Conf Support):** If the email mentions multiple events or entities (e.g., Joint Conferences), do NOT flatten them. Represent them as a **List of Objects** so that no entity's information is lost.\n"
     "4. **Preserve Hierarchy:** If information has a nested relationship (e.g., a specific deadline for a specific track), represent this using nested JSON structures rather than flat text.\n"
     "5. **Data Fidelity:** Do not summarize so much that context is lost. It is better to have a slightly larger nested object than to lose the relationship between a date and its corresponding track.\n"
     "6. **Missing Info:** If a top-level category has no relevant information in the email, set its value to `null`.\n\n"
     
     "Output ONLY the final completed JSON object."),
    ("human",
     "### 1. Target Template Schema (Top-level Anchors):\n"
     "{template_json}\n\n"
     "### 2. Source Email Text:\n"
     "{mail_text}")
])

# 2. 체인 생성
# JsonOutputParser를 사용하여 출력을 바로 딕셔너리로 받습니다.
flexible_chain = json_parser_chain(flexible_summary_prompt)



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