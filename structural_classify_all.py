# process_all.py
from pathlib import Path
import glob
from structure_classify_pipeline.graph import process_one_file, save_json

version = 2
INPUT_DIR = Path("./structure_classify_pipeline/positive_data")
OUT_DIR = Path(f"./structure_classify_pipeline/positive_classify_{version}")  # 결과 저장 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 배치 실행 ===
# 파일명이 숫자.txt 형태(예: 1.txt, 2.txt, ...)만 대상으로
input_files = []
for p in glob.glob(str(INPUT_DIR / "*.json")):
    name = Path(p).stem
    try:
        n = int(name)
    except ValueError:
        continue
    input_files.append((n, Path(p)))

input_files.sort(key=lambda x: x[0])

errors = []
for n, path in input_files:
    try:
        out = process_one_file(path)
        out_name = f"{n}.json"
        save_json(out, OUT_DIR / out_name)
        print(f"[OK] {path.name} -> {out_name}")
    except Exception as e:
        errors.append((path.name, str(e)))
        print(f"[FAIL] {path.name} -> {e}")

# 실패 로그 출력
if errors:
    print("\n=== FAILED FILES ===")
    for name, msg in errors:
        print(f"- {name}: {msg}")
