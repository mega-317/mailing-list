# process_all.py
from pathlib import Path
import glob
from mailflow_pipeline._graph_main import process_one_file, save_json

version = 44
TEXT_DIR = Path("./data/texts")
OUT_DIR = Path(f"./prediction/predictions_{version}")  # 결과 저장 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 배치 실행 ===
# 파일명이 숫자.txt 형태(예: 1.txt, 2.txt, ...)만 대상으로
txt_files = []
for p in glob.glob(str(TEXT_DIR / "*.txt")):
    name = Path(p).stem
    try:
        n = int(name)
    except ValueError:
        continue
    txt_files.append((n, Path(p)))

txt_files.sort(key=lambda x: x[0])

errors = []
for n, path in txt_files:
    try:
        out = process_one_file(path, keep_misspelled_key=True)
        out_name = f"{n}_predict.json"
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
