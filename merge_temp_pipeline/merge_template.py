# 메일 본문을 그냥 요약하도록 지시
from langchain_core.prompts import ChatPromptTemplate
from .common import json_parser_chain
import json

# 병합을 위한 프롬프트 정의
merge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a JSON Schema Integration Specialist. Your goal is to create a comprehensive email summary by using `aligned_json` as the primary structural foundation and augmenting it with unique insights from `free_json`.\n\n"
     
     "## CRITICAL RULES:\n"
     "1. **Preserve Master Structure:** The keys and nesting structure of `aligned_json` are MANDATORY. You must not delete, rename, or alter the original keys from `aligned_json`.\n"
     "2. **Intelligent Augmentation:** Identify valuable information or keys in `free_json` that do not exist in `aligned_json`. Add these as new fields to the root level or appropriate nested objects to ensure a more complete summary.\n"
     "3. **Value Merging:** If both JSONs contain the same information, prioritize the data from `aligned_json`, but enrich it with any specific details found only in `free_json`.\n"
     "4. **No Information Loss:** Ensure that all critical data points (sender, deadlines, specific requests) originally present in `aligned_json` remain fully intact.\n"
     "5. **Conciseness & Optimization:** Even when adding new fields, keep the values pithy and summarized to prevent the JSON size from exploding.\n\n"
     
     "Output ONLY the completed, augmented JSON object without any markdown formatting."),
    ("human",
     "### Master Base Structure (aligned_json):\n"
     "{aligned_json}\n\n"
     "### Source for Augmentation (free_json):\n"
     "{free_json}")
])
merge_chain = json_parser_chain(merge_prompt)

def merge_json_node(state) -> dict:
    """
    두 개의 JSON 요약본을 입력받아 최적화된 하나의 JSON으로 병합합니다.
    """
    # 1. 이전 단계에서 생성된 두 JSON 데이터를 가져옵니다.
    free_json = state.get("free_summary", {})
    aligned_json = state.get("aligned_summary", {})
    
    # 2. LLM 실행 (병합 및 최적화)
    merged_json = merge_chain.invoke({
        "free_json": json.dumps(free_json, indent=2, ensure_ascii=False),
        "aligned_json": json.dumps(aligned_json, indent=2, ensure_ascii=False)
    })
    
    # 3. 최종 결과 반환
    return {
        "merged_summary": merged_json
    }