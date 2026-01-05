import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from .common import MailState
from .merge_template import merge_json_node

graph = StateGraph(MailState)

graph.add_node("merge_summaries", merge_json_node)


graph.add_edge(START, "merge_summaries")
graph.add_edge("merge_summaries", END)

app = graph.compile()

def build_init_state(free_json: dict, aligned_json: dict) -> dict:
    return {
        "free_summary": free_json,
        "aligned_summary": aligned_json,
        "merged_summary": {}
    }

def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def output(result: dict) -> dict:
    # aligned_summary가 있으면 그걸 반환하고, 없으면 빈 딕셔너리 반환
    return result.get("merged_summary") or {}

def process_one_file(free_json_path: Path, aligned_json_path: Path) -> dict:
    """
    두 개의 JSON 파일 경로를 받아 로드한 뒤, 병합된 결과를 반환합니다.
    """
    
    # 1. JSON 파일 로드 로직 추가
    def load_json_safely(path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[Error] JSON 파싱 실패 ({path.name}): {e}")
        return {}

    free_json = load_json_safely(free_json_path)
    aligned_json = load_json_safely(aligned_json_path)
    
    # 2. 입력 유효성 검사
    if not free_json and not aligned_json:
        print("[Warning] 모든 입력 데이터가 비어 있습니다.")
        return {}

    # 3. 상태 초기화
    state = build_init_state(free_json, aligned_json)
    
    # 4. 그래프 실행
    try:
        # 이미 선언된 langgraph app 사용
        result = app.invoke(state)
    except Exception as e:
        print(f"[Error] 그래프 실행 중 오류 발생: {e}")
        return {}
    
    # 5. 최종 결과 반환 (output 유틸리티 사용)
    return output(result)