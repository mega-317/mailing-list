from pathlib import Path
import json
# graph.py에서 수정한 함수들을 임포트
from template_pipeline.graph import process_one_file, save_json

# ------------------------------------------------------------------------------
# file_num = 2
# temp_num = 1


# # --- 경로 설정 ---
# # 1. 입력: 새로운 메일 텍스트 (내용을 채울 소스)
# input_txt_path = Path(f"./data/texts/{file_num}.txt")

# # 2. 참조: 뼈대가 될 기존 JSON (포맷 템플릿)
# reference_json_path = Path(f"./prediction/markdown_1/markdown_{temp_num}_.json")

# # 3. 출력: 결과가 저장될 위치
# output_json_path = Path(f"./prediction/output/fit_{file_num}_to_{temp_num}.json")
# -------------------------------------------------------------------------------


file_num = 53


# --- 경로 설정 ---
# 1. 입력: 새로운 메일 텍스트 (내용을 채울 소스)
input_txt_path = Path(f"./data/texts/{file_num}.txt")

# 2. 참조: 뼈대가 될 기존 JSON (포맷 템플릿)
reference_json_path = Path(f"./prediction/merge/template_v4.json")

# 3. 출력: 결과가 저장될 위치
# output_json_path = Path(f"./prediction/flex_summ/{file_num}.json")
output_json_path = Path(f"./{file_num}.json")




print(f"--- [Start] Structure Transfer ---")
print(f"Input Text: {input_txt_path}")
print(f"Template JSON: {reference_json_path}")

# --- 실행 ---
# process_one_file에 두 개의 경로를 모두 넘겨줍니다.
result_dict = process_one_file(input_txt_path, reference_json_path)

# --- 결과 저장 ---
if result_dict:
    save_json(result_dict, output_json_path)
    print(f"--- [Success] Saved to: {output_json_path}")
else:
    print("--- [Fail] 결과가 비어있습니다. ---")