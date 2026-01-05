import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from .common import MailState
from .template_apply import align_structure_node

graph = StateGraph(MailState)

graph.add_node("align_structure", align_structure_node)


graph.add_edge(START, "align_structure")
graph.add_edge("align_structure", END)

app = graph.compile()

def build_init_state(mail_text: str, template: dict = None) -> dict:
    return {
        "mail_text": mail_text,
        "aligned_summary": None,
        "template": template # 참조용 JSON을 상태에 주입
    }

def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def output(result: dict) -> dict:
    # aligned_summary가 있으면 그걸 반환하고, 없으면 빈 딕셔너리 반환
    return result.get("aligned_summary") or {}

# [핵심 수정] ref_json_path 인자 추가
def process_one_file(txt_path: Path, ref_json_path: Path = None) -> dict:
    # 1. 메일 텍스트 읽기
    mail_text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    
    # 2. 참조용 JSON 파일 읽기 (있다면)
    reference_data = {}
    if ref_json_path and ref_json_path.exists():
        try:
            json_text = ref_json_path.read_text(encoding="utf-8")
            reference_data = json.loads(json_text)
        except json.JSONDecodeError:
            print(f"[Error] 참조 JSON 파일 파싱 실패: {ref_json_path}")
    
    # 3. 상태 초기화 (참조 데이터 포함)
    state = build_init_state(mail_text, reference_data)
    
    # 4. 그래프 실행
    result = app.invoke(state)
    
    # 5. 결과 반환 (dict 형태)
    return output(result)