import os
import json

OUTPUT_FOLDER = '.'
result = {
  "is_call_for_paper": None,
  "conference_name": None,
  "start_date": None,
  "submission_deadline": None,
  "conference_website": None,
  "special_note": None
}

# 🔁 텍스트 파일 순회 및 처리
for i in range(1, 101):
    output_path = os.path.join(OUTPUT_FOLDER, f"{i}_label.json")
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, ensure_ascii=False, indent=2)
    print(f"✅ Saved: {output_path}")
