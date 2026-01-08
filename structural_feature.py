import json
import os
from collections import Counter

def get_structural_features(data, prefix='', features=None):
    """JSON 트리에서 모든 계층 경로(Paths)와 리프 키(Leaf Keys)를 추출"""
    if features is None:
        features = {"paths": set(), "leaf_keys": set()}
    
    if isinstance(data, dict):
        for k, v in data.items():
            # 키 이름 정규화: 소문자화 및 공백/숫자 제거로 LLM 변이 최소화
            norm_k = "".join([i for i in k.lower() if i.isalpha() or i == '_'])
            path = f"{prefix}/{norm_k}" if prefix else norm_k
            
            features["paths"].add(path)
            
            if not isinstance(v, (dict, list)):
                features["leaf_keys"].add(norm_k)
            else:
                get_structural_features(v, path, features)
    elif isinstance(data, list):
        for item in data:
            get_structural_features(item, prefix, features)
            
    return features

def analyze_specific_jsons(base_dir, file_numbers, threshold=0.6):
    """
    지정된 번호의 JSON 파일들을 분석하여 공통 구조 추출
    - threshold=1.0: 10개 파일 모두에 존재하는 구조만 추출
    """
    json_data_list = []
    
    # 1. 파일 로드
    for num in file_numbers:
        file_path = os.path.join(base_dir, f"{num}.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data_list.append(json.load(f))
        except FileNotFoundError:
            print(f"경고: {file_path} 파일을 찾을 수 없습니다.")

    if not json_data_list:
        return None, None

    # 2. 구조 특징 수집
    total_files = len(json_data_list)
    path_counts = Counter()
    leaf_counts = Counter()
    
    for data in json_data_list:
        features = get_structural_features(data)
        path_counts.update(features["paths"])
        leaf_counts.update(features["leaf_keys"])
    
    # 3. 임계값에 따른 공통 특징 필터링
    common_paths = sorted([k for k, v in path_counts.items() if (v / total_files) >= threshold])
    common_leaves = sorted([k for k, v in leaf_counts.items() if (v / total_files) >= threshold])

    return common_paths, common_leaves

# --- 설정 및 실행 ---
# 파일이 들어있는 디렉토리 경로
BASE_DIRECTORY = "./structure_classify_pipeline/positive_data/"

# 분석하고 싶은 10개의 파일 번호를 아래 리스트에 입력해주세요.
target_numbers = [2, 11, 16, 28, 32, 33, 51, 61, 93, 97] # 예시 번호입니다.

paths, leaves = analyze_specific_jsons(BASE_DIRECTORY, target_numbers)

if paths:
    print(f"\n[분석 결과: {len(target_numbers)}개 파일의 공통 구조]")
    print(f"\n1. 100% 공통된 계층 경로 (Total: {len(paths)}개):")
    for p in paths:
        print(f"  - {p}")
        
    print(f"\n2. 100% 공통된 최하위 키 (Total: {len(leaves)}개):")
    print(f"  {', '.join(leaves)}")
else:
    print("분석할 데이터가 없습니다.")