import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from .common import MailState
from .entity_restructuring import entity_node

graph = StateGraph(MailState)

graph.add_node("entity", entity_node)


graph.add_edge(START, "entity")
graph.add_edge("entity", END)

app = graph.compile()

def build_init_state(mail_text: str) -> dict:
    return {
        "mail_text": mail_text,
        "result": None
    }

def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def output(result: dict) -> dict:
    return result.get("result") or {}

# [핵심 수정] ref_json_path 인자 추가
def process_one_file(txt_path: Path) -> dict:

    mail_text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")

    # 3. 상태 초기화 (참조 데이터 포함)
    state = build_init_state(mail_text)
    
    # 4. 그래프 실행
    result = app.invoke(state)
    
    # 5. 결과 반환 (dict 형태)
    return output(result)