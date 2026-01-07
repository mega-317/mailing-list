import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from .common import MailState
from .structural_classify import validation_node

graph = StateGraph(MailState)

graph.add_node("validation", validation_node)


graph.add_edge(START, "validation")
graph.add_edge("validation", END)

app = graph.compile()

def build_init_state(reference_data: dict = None) -> dict:
    return {
        "input_json": reference_data,
        "mail_type": None
    }

def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def output(result: dict) -> dict:
    return result.get("mail_type") or {}

# [핵심 수정] ref_json_path 인자 추가
def process_one_file(ref_json_path: Path = None) -> dict:

    # 2. 참조용 JSON 파일 읽기 (있다면)
    reference_data = {}
    if ref_json_path and ref_json_path.exists():
        try:
            json_text = ref_json_path.read_text(encoding="utf-8")
            reference_data = json.loads(json_text)
        except json.JSONDecodeError:
            print(f"[Error] 참조 JSON 파일 파싱 실패: {ref_json_path}")
    
    # 3. 상태 초기화 (참조 데이터 포함)
    state = build_init_state(reference_data)
    
    # 4. 그래프 실행
    result = app.invoke(state)
    
    # 5. 결과 반환 (dict 형태)
    return output(result)