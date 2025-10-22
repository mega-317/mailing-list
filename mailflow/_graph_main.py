# mailflow/graph_main.py
from pathlib import Path
import json
from langgraph.graph import StateGraph, START, END
from .common import MailState
from ._1_check_body import check_mail_body_node
from ._2_extract_cfp import extract_text_node
from ._3_summary import summarize
from ._4_determine_cfp import cfp_candidate_node, classify_cfp_target_node
from ._5_joint import is_joint_call_node, is_joint_conf_node
from ._6_info import harvest_infos_node, finalize_infos_text_node
from ._7_names import ext_conf_name_node, ext_work_name_node, build_conf_tokens_node, final_conf_name_node
from ._8_dates import ext_start_date_node, ext_submission_deadline_node, submission_deadline_candidates_node, final_submission_deadline_node
from ._9_url import ext_conf_url_node

# --- Routers ---
def cfp_candidate_router(state: MailState) -> str:
    return "go_text" if state.get("cfp_candidate", False) else "end"

def check_mail_body_router(state: MailState) -> str:
    return "go_next" if state.get("has_body", True) else "end"

def classify_cfp_router(state: MailState) -> str:
    return "go_next" if state.get("is_cfp", False) else "end"

def is_joint_call_router(state: MailState) -> str:
    return "joint_call" if state.get("is_joint_call", False) else "go_next"

def is_joint_conf_router(state: MailState) -> str:
    return "end" if state.get("is_joint_conf", False) else "go_next"

# --- Graph build ---
graph = StateGraph(MailState)
graph.add_node("check_mail_body", check_mail_body_node)
graph.add_node("summarize", summarize)
graph.add_node("extract_text", extract_text_node)
graph.add_node("cfp_candidate", cfp_candidate_node)
graph.add_node("classify_cfp_target", classify_cfp_target_node)
graph.add_node("is_joint_call", is_joint_call_node)
graph.add_node("is_joint_conf", is_joint_conf_node)
graph.add_node("harvest_infos", harvest_infos_node)
graph.add_node("finalize_infos_text", finalize_infos_text_node)
graph.add_node("ext_conf_name", ext_conf_name_node)
graph.add_node("ext_work_name", ext_work_name_node)
graph.add_node("build_conf_tokens", build_conf_tokens_node)
graph.add_node("final_conf_name", final_conf_name_node)
graph.add_node("ext_start_date", ext_start_date_node)
graph.add_node("ext_submission_deadline", ext_submission_deadline_node)
graph.add_node("submission_deadline_candidates", submission_deadline_candidates_node)
graph.add_node("final_submission_deadline", final_submission_deadline_node)
graph.add_node("ext_conf_url", ext_conf_url_node)


graph.add_edge(START, "check_mail_body")
graph.add_conditional_edges(
    "check_mail_body",
    check_mail_body_router,
    {
        "go_next": "cfp_candidate",
        "end": END
    }
)
graph.add_conditional_edges(
    "cfp_candidate",
    cfp_candidate_router,
    {
        "go_text": "classify_cfp_target",
        "end": END
    }
)
graph.add_conditional_edges(
    "classify_cfp_target",
    classify_cfp_router,
    {
        "go_next": "is_joint_call",
        "end": END
    }
)
graph.add_conditional_edges(
    "is_joint_call",    
    is_joint_call_router,
    {
        "joint_call": "is_joint_conf",
        "go_next": "ext_conf_name",
    }
)
graph.add_conditional_edges(
    "is_joint_conf",
    is_joint_conf_router,
    {
        "go_next": "ext_conf_name",
        "end": END
    }
)
graph.add_edge("ext_conf_name", "build_conf_tokens")
graph.add_edge("build_conf_tokens", "final_conf_name")
graph.add_edge("final_conf_name", "ext_start_date")
graph.add_edge("ext_start_date", "submission_deadline_candidates")
graph.add_edge("submission_deadline_candidates", "final_submission_deadline")
graph.add_edge("final_submission_deadline", "ext_conf_url")
graph.add_edge("ext_conf_url", END)

# graph.add_edge(START, "check_mail_body")
# graph.add_conditional_edges(
#     "check_mail_body",
#     check_mail_body_router,
#     {
#         "go_next": "extract_text",
#         "end": END
#     }
# )
# graph.add_edge("extract_text", "summarize")
# graph.add_edge("summarize", "cfp_candidate")
# graph.add_conditional_edges(
#     "cfp_candidate",
#     cfp_candidate_router,
#     {
#         "go_text": "classify_cfp_target",
#         "end": END
#     }
# )
# graph.add_conditional_edges(
#     "classify_cfp_target",
#     classify_cfp_router,
#     {
#         "go_next": "is_joint_conf",
#         "end": END
#     }
# )
# graph.add_conditional_edges(
#     "is_joint_conf",
#     is_joint_conf_router,
#     {
#         "go_next": "is_joint_work",
#         "end": END
#     }
# )
# # graph.add_edge("is_joint_work", END)
# graph.add_edge("is_joint_work", "harvest_infos")
# graph.add_edge("harvest_infos", "finalize_infos_text")
# graph.add_edge("finalize_infos_text", "ext_conf_name")
# graph.add_edge("ext_conf_name", "build_conf_tokens")
# graph.add_edge("build_conf_tokens", "final_conf_name")
# graph.add_edge("final_conf_name", "ext_start_date")
# graph.add_edge("ext_start_date", "submission_deadline_candidates")
# graph.add_edge("submission_deadline_candidates", "final_submission_deadline")
# graph.add_edge("final_submission_deadline", "ext_conf_url")
# graph.add_edge("ext_conf_url", END)

app = graph.compile()


# --- Helpers ---
def build_init_state(mail_text: str) -> dict:
    return {
        "mail_text": mail_text,
        "cfp_candidate": None,
        "purpose": None,
        "extracted_text": None,
        "classify_cfp_target": None,
        "is_cfp": False,
        "is_cfp_final": None,
        "has_body": True,
        "is_joint_call": False,
        "is_joint_conf": False,
        "infos": [],
        "infos_text": None,
        "evidence_sentences": None,
        "conf_name_candidates": [],
        "conf_name_final": None,
        "conf_tokens": [],
        "work_name_candidates": [],
        "work_name_tokens": [],
        "start_date": None,
        "sub_deadline": None,
        "sub_deadline_candidate": [],
        "conf_website": None,
        "len_mail_text": 0,
        "len_purpose": 0,
        "len_extracted_text": 0,
        "len_infos_text": 0
    }

def normalize_output(result: dict, keep_misspelled_key: bool = True) -> dict:
    return {
        "has_body": result.get("has_body"),
        "purpose": result.get("purpose"),
        "extracted_text": result.get("extracted_text"),
        "cfp": {
            "cfp_candidate": result.get("cfp_candidate"),
            "classify_cfp_target": result.get("classify_cfp_target"),
            "is_cfp": result.get("is_cfp"),
        },
        "is_cfp_final": result.get("is_cfp_final"),
        "is_joint_call": {
            "is_joint_call": result.get("is_joint_call"),
            "is_joint_conf": result.get("is_joint_conf"),
        },
        "infos": {
            "infos": result.get("infos"),
            "conf_name_candidates": result.get("conf_name_candidates"),
            "conf_name_tokens": result.get("conf_tokens"),
            "submission_deadline_candidates": result.get("sub_deadline_candidate"),
        },
        "length": {
            "mail_text": result.get("len_mail_text"),
            "extracted_text": result.get("len_extracted_text"),
            "infos": result.get("len_infos_text")
        },
        "conference_name": result.get("conf_name_final"),
        "start_date": result.get("start_date"),
        "submission_deadline": result.get("sub_deadline"),   
        "conference_website": result.get("conf_website"),
    }

def process_one_file(txt_path: Path, keep_misspelled_key: bool = True) -> dict:
    mail_text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    state = build_init_state(mail_text)
    result = app.invoke(state)
    return normalize_output(result, keep_misspelled_key)

def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")