import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

def flatten_to_tokens(data, prefix=''):
    """
    JSON 구조를 '경로'와 '개별 키' 토큰으로 변환합니다.
    시맨틱 정규화(유의어 처리)를 포함하여 의미적 유사도를 반영합니다.
    """
    tokens = []
    # 1. 시맨틱 정규화 매핑 (유의어 사전)
    synonyms = {
        'venue': 'location', 'place': 'location', 'site': 'location',
        'url': 'website', 'link': 'website',
        'deadline': 'date', 'due': 'date', 'time': 'date',
        'track': 'area', 'topic': 'area'
    }

    if isinstance(data, dict):
        for k, v in data.items():
            # 키 이름 정규화
            clean_k = "".join([c.lower() for c in k if c.isalpha() or c == '_'])
            for syn, target in synonyms.items():
                if syn in clean_k:
                    clean_k = clean_k.replace(syn, target)

            # 계층 경로 생성 (구조 반영)
            path = f"{prefix}/{clean_k}" if prefix else clean_k
            
            tokens.append(path)      # 전체 경로 추가
            tokens.append(clean_k)   # 개별 키 추가 (의미 반영)
            
            if isinstance(v, (dict, list)):
                tokens.extend(flatten_to_tokens(v, path))
    elif isinstance(data, list):
        for item in data:
            tokens.extend(flatten_to_tokens(item, prefix))
    return tokens

def load_data(pos_dir, neg_dir):
    """지정된 폴더에서 데이터를 로드하여 텍스트 뭉치와 라벨 생성"""
    documents = []
    labels = []
    filenames = []

    configs = [(pos_dir, 1, 'Positive'), (neg_dir, 0, 'Negative')]
    
    for directory, label, name in configs:
        if not os.path.exists(directory):
            print(f"경고: {directory} 폴더가 없습니다.")
            continue
            
        for f in os.listdir(directory):
            if f.endswith('.json'):
                path = os.path.join(directory, f)
                with open(path, 'r', encoding='utf-8') as file:
                    try:
                        data = json.load(file)
                        # JSON을 토큰화하여 문장 형태로 변환
                        doc_text = " ".join(flatten_to_tokens(data))
                        documents.append(doc_text)
                        labels.append(label)
                        filenames.append(f)
                    except Exception as e:
                        print(f"파일 오류 ({f}): {e}")
                        
    return documents, np.array(labels), filenames

# --- 설정 및 실행 ---
# 1. 경로 설정
POSITIVE_PATH = './structure_classify_pipeline/positive_data/'
NEGATIVE_PATH = './structure_classify_pipeline/negative_data/'

# 2. 데이터 처리
print("데이터를 로드하고 구조 분석을 시작합니다...")
docs, y, names = load_data(POSITIVE_PATH, NEGATIVE_PATH)

if len(docs) < 2:
    print("분석할 데이터가 충분하지 않습니다.")
else:
    # 3. TF-IDF 벡터화
    # token_pattern을 지정하여 슬래시(/)가 포함된 경로를 하나의 단어로 인식하게 함
    vectorizer = TfidfVectorizer(token_pattern=r'[^\s]+', binary=True)
    X = vectorizer.fit_transform(docs).toarray()

    # 4. t-SNE 알고리즘을 통한 2차원 투영
    # 데이터 개수에 따라 perplexity 조절 (데이터가 적으면 낮게 설정)
    ppl = min(30, len(docs) - 1)
    tsne = TSNE(n_components=2, perplexity=ppl, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)

    # 5. 시각화
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red']
    target_names = ['Non-CFP (Negative)', 'CFP (Positive)']

    for i, color, label_name in zip([0, 1], colors, target_names):
        idx = np.where(y == i)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=color, label=label_name, 
                    s=120, alpha=0.6, edgecolors='k')

    # 데이터 포인트 위에 파일명 표시 (샘플링)
    for i, txt in enumerate(names):
        if i % max(1, len(names)//15) == 0: # 너무 많으면 복잡하므로 일부만 표시
            plt.annotate(txt, (X_2d[i, 0], X_2d[i, 1]), xytext=(5, 5), 
                         textcoords='offset points', fontsize=9, alpha=0.8)

    plt.title("Visualizing JSON Structural & Semantic Clustering (t-SNE)")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 결과 저장
    plt.savefig("json_structural_clustering.png")
    print(f"분석 완료! 시각화 결과가 'json_structural_clustering.png'로 저장되었습니다.")
    print(f"총 분석 파일 수: {len(docs)}, 추출된 특징(Feature) 수: {X.shape[1]}")