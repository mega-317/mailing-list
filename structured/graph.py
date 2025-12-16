import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from .common import MailState
from .summary_string import summ_str_node
from .summary_markdown import summ_mark_node
from .reorganize import reorganize_node
from .structure_transfer import align_structure_node

graph = StateGraph(MailState)
graph.add_node("summ_str", summ_str_node)
graph.add_node("summ_mark", summ_mark_node)
graph.add_node("reorganize", reorganize_node)
graph.add_node("align_structure", align_structure_node)


graph.add_edge(START, "align_structure")
graph.add_edge("align_structure", END)

app = graph.compile()

def build_init_state(mail_text: str) -> dict:
    return {
        "mail_text": mail_text,
        "summary": None,
        "summary_dict": None,
        "aligned_summary": None,
        "reference_summary_json": None
    }

# json으로 저장하는 파트
# def output(result: dict) -> dict:
#     return {
#         "summary": result.get("summary")
#     }

def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# [수정됨] 딕셔너리가 아닌 '문자열'을 반환하도록 변경
def output(result: dict) -> str:
    # summary 값이 없으면 빈 문자열 반환
    return result.get("aligned_summary") or ""

# [수정됨] JSON 덤프 과정 없이 텍스트를 그대로 파일에 쓰도록 변경
def save_text(content: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # content가 문자열인지 확인 후 저장
    if content is None:
        content = ""
    out_path.write_text(str(content), encoding="utf-8")

def process_one_file(txt_path: Path) -> str:
    mail_text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    state = build_init_state(mail_text)
    result = app.invoke(state)
    return output(result)