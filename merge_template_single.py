from pathlib import Path
import json
# graph.py에서 수정한 함수들을 임포트
from merge_temp_pipeline.graph import process_one_file, save_json


# -------------------------------------------------------------------------------------------------
# LLM_summ_json = 3
# temp_summ_json = 1

# # LLM이 스스로 요약한 메일
# LLM_summ_json_path = Path(f"./prediction/markdown_1/markdown_{LLM_summ_json}_.json")

# # 템플릿에 맞게 요약한 메일
# temp_summ_json_path = Path(f"./prediction/output/fit_{LLM_summ_json}_to_{temp_summ_json}.json")

# # 3. 출력: 결과가 저장될 위치
# output_json_path = Path(f"./prediction/merge/merge_template_{LLM_summ_json}_{temp_summ_json}.json")
# -------------------------------------------------------------------------------------------------


LLM_summ_json = 7

# LLM이 스스로 요약한 메일
LLM_summ_json_path = Path(f"./prediction/markdown_1/markdown_{LLM_summ_json}_.json")

# 템플릿에 맞게 요약한 메일
temp_summ_json_path = Path(f"./prediction/output/fit_{LLM_summ_json}.json")

# 3. 출력: 결과가 저장될 위치
output_json_path = Path(f"./prediction/merge/merge_template_{LLM_summ_json}.json")





print(f"--- [Start] JSON MERGE ---")
print(f"First JSON: {LLM_summ_json_path}")
print(f"Second JSON: {temp_summ_json_path}")

# --- 실행 ---
# process_one_file에 두 개의 경로를 모두 넘겨줍니다.
result_dict = process_one_file(LLM_summ_json_path, temp_summ_json_path)

# --- 결과 저장 ---
if result_dict:
    save_json(result_dict, output_json_path)
    print(f"--- [Success] Saved to: {output_json_path}")
else:
    print("--- [Fail] 결과가 비어있습니다. ---")