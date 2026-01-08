import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer, util

# 1. 시먼틱 모델 로드 (의미 비교용 AI)
# all-MiniLM-L6-v2는 가볍고 속도가 빨라 키 이름 비교에 최적화되어 있습니다.
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_all_unique_keys(data):
    """JSON 데이터에서 사용된 모든 고유 키(Key) 이름을 수집합니다."""
    keys = set()
    if isinstance(data, dict):
        for k, v in data.items():
            # 키 이름 정규화 (특수문자 제거 및 소문자화)
            clean_k = "".join([c.lower() for c in k if c.isalpha() or c == '_'])
            keys.add(clean_k)
            keys.update(get_all_unique_keys(v))
    elif isinstance(data, list):
        for item in data:
            keys.update(get_all_unique_keys(item))
    return keys

def build_semantic_map(unique_keys, threshold=0.75):
    """
    SBERT를 사용하여 유사한 의미의 키들을 자동으로 그룹화합니다.
    예: 'venue', 'location', 'place' -> 'location'으로 통일
    """
    if not unique_keys: return {}
    
    keys_list = list(unique_keys)
    # 단어들을 벡터로 변환 (Embedding)
    embeddings = embed_model.encode(keys_list, convert_to_tensor=True)
    
    # 모든 단어 쌍 간의 코사인 유사도 계산
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    semantic_map = {}
    used_keys = set()
    
    for i in range(len(keys_list)):
        if keys_list[i] in used_keys: continue
        
        # 현재 키와 유사도가 threshold(0.75) 이상인 키들을 찾음
        similar_indices = (cosine_scores[i] >= threshold).nonzero().flatten().tolist()
        
        # 해당 그룹의 대표 키 선정 (가장 짧은 단어를 표준어로 사용)
        group = [keys_list[idx] for idx in similar_indices]
        representative = min(group, key=len)
        
        for k in group:
            semantic_map[k] = representative
            used_keys.add(k)
            
    return semantic_map

def flatten_to_tokens_sbert(data, semantic_map, prefix=''):
    """자동 생성된 semantic_map을 사용하여 JSON 구조를 토큰화합니다."""
    tokens = []
    if isinstance(data, dict):
        for k, v in data.items():
            clean_k = "".join([c.lower() for c in k if c.isalpha() or c == '_'])
            
            # AI가 판별한 표준 키로 변환 (없으면 원래 키 유지)
            normalized_k = semantic_map.get(clean_k, clean_k)
            
            # 계층 경로 생성
            path = f"{prefix}/{normalized_k}" if prefix else normalized_k
            
            tokens.append(path)      # 구조(경로) 토큰
            tokens.append(normalized_k) # 의미(키) 토큰
            
            if isinstance(v, (dict, list)):
                tokens.extend(flatten_to_tokens_sbert(v, semantic_map, path))
    elif isinstance(data, list):
        for item in data:
            tokens.extend(flatten_to_tokens_sbert(item, semantic_map, prefix))
    return tokens

# --- 실행 파이프라인 ---

POSITIVE_PATH = './structure_classify_pipeline/positive_data/'
NEGATIVE_PATH = './structure_classify_pipeline/negative_data/'

# 1. 모든 파일의 고유 키 먼저 수집
print("1단계: 데이터 전수 조사를 통한 고유 키 수집 중...")
all_keys = set()
for folder in [POSITIVE_PATH, NEGATIVE_PATH]:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.json'):
                with open(os.path.join(folder, f), 'r', encoding='utf-8') as file:
                    all_keys.update(get_all_unique_keys(json.load(file)))

# 2. SBERT 기반 자동 유의어 맵 생성
print(f"2단계: AI 모델(SBERT)을 이용한 키 의미 분석 중... (총 {len(all_keys)}개 키)")
semantic_map = build_semantic_map(all_keys, threshold=0.75)

# 3. 데이터 로드 및 토큰화
print("3단계: 자동 정규화된 경로 데이터 생성 중...")
documents = []
labels = []
filenames = []

for folder, label in [(POSITIVE_PATH, 1), (NEGATIVE_PATH, 0)]:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.json'):
                with open(os.path.join(folder, f), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    doc_text = " ".join(flatten_to_tokens_sbert(data, semantic_map))
                    documents.append(doc_text)
                    labels.append(label)
                    filenames.append(f)

# 4. TF-IDF 및 t-SNE 시각화 (기존 로직과 동일)
print("4단계: TF-IDF 가중치 계산 및 t-SNE 차원 축소 중...")
vectorizer = TfidfVectorizer(token_pattern=r'[^\s]+', binary=True)
X = vectorizer.fit_transform(documents).toarray()

tsne = TSNE(n_components=2, perplexity=min(30, len(documents)-1), random_state=42)
X_2d = tsne.fit_transform(X)

# 5. 시각화 출력
plt.figure(figsize=(12, 8))
plt.scatter(X_2d[np.array(labels)==1, 0], X_2d[np.array(labels)==1, 1], c='red', label='CFP (Positive)', alpha=0.6)
plt.scatter(X_2d[np.array(labels)==0, 0], X_2d[np.array(labels)==0, 1], c='blue', label='Non-CFP (Negative)', alpha=0.6)
plt.title("SBERT-Enhanced Structural Clustering")
plt.legend()
plt.savefig("sbert_structural_clustering.png")
print("모든 공정이 완료되었습니다. 'sbert_structural_clustering.png'를 확인하세요.")