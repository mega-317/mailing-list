from pathlib import Path
import json
# graph.py에서 수정한 함수들을 임포트
from entity_pipeline.graph import process_one_file, save_json

file_num = 94


# --- 경로 설정 ---
# 1. 입력: 새로운 메일 텍스트 (내용을 채울 소스)
input_path = Path(f"./data/texts/{file_num}.txt")

# 3. 출력: 결과가 저장될 위치
output_path = Path(f"./prediction/restructuring_1/{file_num}.json")




print(f"--- [Start] Structure Transfer ---")
print(f"Input Text: {input_path}")

# --- 실행 ---
# process_one_file에 두 개의 경로를 모두 넘겨줍니다.
result_dict = process_one_file(input_path)

# --- 결과 저장 ---
if result_dict:
    save_json(result_dict, output_path)
    print(f"--- [Success] Saved to: {output_path}")
else:
    print("--- [Fail] 결과가 비어있습니다. ---")