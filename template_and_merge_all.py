from pathlib import Path
import json

from template_process_single import process_one_file as run_structure_transfer
from merge_template_single import process_one_file as run_merge_process
from template_pipeline.graph import save_json

def run_total_pipeline(file_num: int):
    print(f"\n==== [Step 0] Initialization for File No. {file_num} ====")
    
    # --- 경로 설정 ---
    input_txt_path = Path(f"./data/texts/{file_num}.txt")
    base_template_path = Path(f"./prediction/merge/template.json")
    
    # 1단계 결과물 (구조화 요약) 경로
    fit_output_path = Path(f"./prediction/output/fit_{file_num}.json")
    
    # 2단계를 위한 자유 요약본 경로
    free_summary_path = Path(f"./prediction/markdown_1/markdown_{file_num}_.json")
    
    # [수정] 최종 결과물 저장 경로 (두 가지 버전)
    master_template_path = Path(f"./prediction/merge/template.json") # 계속 덮어씌워짐 (Master)
    history_template_path = Path(f"./prediction/merge/merge_template_{file_num}.json") # 번호별 저장 (Snapshot)

    # --------------------------------------------------------------------------
    # STEP 1: Structure Transfer (텍스트 -> 템플릿에 맞춘 요약)
    # --------------------------------------------------------------------------
    print(f"\n {file_num}번 메일을 템플릿에 맞춰 요약")
    
    aligned_result = run_structure_transfer(input_txt_path, base_template_path)
    
    if not aligned_result:
        print(f"!!! [Fail] Step 1 결과가 비어있습니다. (File {file_num})")
        return

    save_json(aligned_result, fit_output_path)
    print(f"---- [Success] {fit_output_path.name}에 저장됨")

    # --------------------------------------------------------------------------
    # STEP 2: Intelligent Merge (자유 요약 + 구조화 요약 병합)
    # --------------------------------------------------------------------------
    print(f"\n>템플릿 병합 시작")

    merged_result = run_merge_process(free_summary_path, fit_output_path)

    if not merged_result:
        print(f"!!! [Fail] Step 2 결과가 비어있습니다. (File {file_num})")
        return

    # [핵심 수정] 두 군데에 모두 저장
    # 1. 마스터 템플릿 업데이트 (다음 루프의 base_template으로 활용됨)
    save_json(merged_result, master_template_path)
    
    # 2. 개별 스냅샷 저장 (변화 과정 확인용)
    save_json(merged_result, history_template_path)
    
    print(f"[Success]")
    print(f"Updated Master: {master_template_path.name}")
    print(f"Saved Snapshot: {history_template_path.name}")

if __name__ == "__main__":
    # 8부터 100까지 반복 실행
    for i in range(80, 101):
        print(f"\n" + "="*60)
        print(f">>> Processing File Number: {i}")
        print("="*60)
        
        try:
            run_total_pipeline(i)
        except Exception as e:
            print(f"!!! [Critical Error] File {i} failed: {e}")
            continue
            
    print("\n" + "★"*25)
    print("모든 파이프라인 작업이 완료되었습니다!")
    print("★"*25)