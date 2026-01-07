# process_one_file.py
from pathlib import Path
from structured_pipeline.graph import process_one_file, save_text, save_json

# json으로 결과 저장
# --- 여기서 단일 파일 실행 ---
input_path = Path("./data/texts/2.txt")           # 입력 파일
output_path = Path("./structuring_2.json")   # 출력 파일

out = process_one_file(input_path)
save_json(out, output_path)

# 잘 됐는지 출력
print(f"[OK] {input_path.name} -> {output_path.name}")


