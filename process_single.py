# process_one_file.py
from pathlib import Path
from mailflow._graph_main import process_one_file, save_json

# --- 여기서 단일 파일 실행 ---
input_path = Path("./data/texts/5.txt")           # 입력 파일
output_path = Path("result/single_predict.json")   # 출력 파일

out = process_one_file(input_path, keep_misspelled_key=True)
save_json(out, output_path)

print(f"[OK] {input_path.name} -> {output_path.name}")
