import os
import json
from datetime import datetime
from statistics import mean, median

version = 1
label_dir = './data/seworld_label'
predict_dir = f'./prediction/validate'

fields_to_compare = [
    "conference_name",
    "start_date",
    "submission_deadline",
    "conference_website"
]

def parse_date(s):
    if not s:
        return None
    try:
        # ISO YYYY-MM-DD 가정
        return datetime.fromisoformat(s).date()
    except Exception:
        return None

# 집계용
match_count = 0
comparison_results = []

# 필드별 집계
per_field = {
    # 이진 분류 전용 (라벨 vs 예측)
    "is_call_for_paper": {
        "total": 0, "matches": 0, "errors": 0,
        "tp": 0, "fp": 0, "tn": 0, "fn": 0
    }
}
for f in fields_to_compare:
    per_field[f] = {
        "total": 0, "matches": 0, "errors": 0,
        "null_to_value": 0, "value_to_null": 0, "wrong_value": 0,
        # 날짜용 지표
        "abs_day_errors": []  # 둘 다 값이 있고 불일치인 경우 |days| 누적
    }

# 비교 루프
for i in range(1, 101):
    label_path = os.path.join(label_dir, f"{i}_label.json")
    predict_path = os.path.join(predict_dir, f"{i}.json")

    with open(label_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    with open(predict_path, 'r', encoding='utf-8') as f:
        predict_data = json.load(f)

    mismatched_fields = []

    # --- 이진: is_call_for_paper vs is_call_for_conference_paper ---
    y = label_data.get("is_call_for_paper")
    yhat = predict_data.get("is_valid_cfp")
    per_field["is_call_for_paper"]["total"] += 1
    if y == yhat:
        per_field["is_call_for_paper"]["matches"] += 1
    else:
        per_field["is_call_for_paper"]["errors"] += 1
        mismatched_fields.append({
            "field": "is_cfp",
            "label_value": y,
            "predict_value": yhat,
        })
        
    # 혼동행렬
    if y is True and yhat is True:
        per_field["is_call_for_paper"]["tp"] += 1
    elif y is False and yhat is True:
        per_field["is_call_for_paper"]["fp"] += 1
    elif y is False and yhat is False:
        per_field["is_call_for_paper"]["tn"] += 1
    elif y is True and yhat is False:
        per_field["is_call_for_paper"]["fn"] += 1
        

    # --- 일반 필드 ---
    # for field in fields_to_compare:
    #     # label_value = label_data.get("conference_name")
    #     # predict_value = predict_data.get("infos", {}).get("conf_name_final")
        
    #     label_value = label_data.get(field)
    #     predict_value = predict_data.get(field)
    #     # predict_value = predict_data.get("infos")
    #     per_field[field]["total"] += 1

    #     if label_value == predict_value:
    #         per_field[field]["matches"] += 1
    #         continue

    #     # 불일치 기록
    #     per_field[field]["errors"] += 1
    #     if label_value is None and predict_value is not None:
    #         per_field[field]["null_to_value"] += 1
    #     elif label_value is not None and predict_value is None:
    #         per_field[field]["value_to_null"] += 1
    #     else:
    #         per_field[field]["wrong_value"] += 1

    #     # 상세 목록에 추가
    #     mismatched_fields.append({
    #         "field": field,
    #         "label_value": label_value,
    #         "predict_value": predict_value
    #     })

    if mismatched_fields:
        comparison_results.append({
            "file": f"{i}_predict.json",
            "mismatches": mismatched_fields
        })
    else:
        match_count += 1

# --- 요약 메트릭 계산 ---
def safe_div(n, d):
    return n / d if d else 0.0

# is_call_for_paper 메트릭
cm = per_field["is_call_for_paper"]
precision = safe_div(cm["tp"], (cm["tp"] + cm["fp"]))
recall = safe_div(cm["tp"], (cm["tp"] + cm["fn"]))
f1 = safe_div(2 * precision * recall, (precision + recall))
binary_accuracy = safe_div(cm["matches"], cm["total"])

# 필드별 정확도 및 날짜 오차 요약
field_summaries = {}
for f in fields_to_compare:
    data = per_field[f]
    acc = safe_div(data["matches"], data["total"])
    mae = mean(data["abs_day_errors"]) if data["abs_day_errors"] else None
    medae = median(data["abs_day_errors"]) if data["abs_day_errors"] else None
    field_summaries[f] = {
        "matches": data["matches"],
        "errors": data["errors"],
        "null_to_value": data["null_to_value"],
        "value_to_null": data["value_to_null"],
        "wrong_value": data["wrong_value"],
    }

summary = {
    "overall": {
        "total_files": 100,
        "files_all_correct": match_count,
        "files_with_any_error": len(comparison_results),
    },
    "per_field": {
        "is_call_for_paper": {
            "matches": cm["matches"],
            "errors": cm["errors"],
            "label_true_predict_false": cm["fn"],  # 실제 True, 예측 False
            "label_false_predict_true": cm["fp"],  # 실제 False, 예측 True
        },
        **field_summaries
    }
}

# --- 출력 및 저장 ---
print(f"총 {match_count}개의 파일이 전부 일치합니다.")
for item in comparison_results:
    print(f"\n❌ {item['file']} 불일치:")
    for mismatch in item['mismatches']:
        print(f"  - 필드: {mismatch['field']}")
        print(f"    라벨 값: {mismatch['label_value']}")
        print(f"    예측 값: {mismatch['predict_value']}")

out_path = f"result/comparison_result_{version}.json"
with open(out_path, "w", encoding="utf-8") as out_file:
    json.dump({
        "summary": summary,
        "match_count": match_count,
        "total_files": 100,
        "mismatched_files": comparison_results
    }, out_file, ensure_ascii=False, indent=2)

print(f"\n📁 결과가 '{out_path}'에 저장되었습니다.")
