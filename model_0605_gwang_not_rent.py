import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances
import joblib

# 1. 데이터 불러오기
df = pd.read_csv("cleaned_gwangju_not_rent.csv", encoding='utf-8-sig')
df = df[['dong', 'exclusive_area', 'price']].dropna()

# 2. 전처리 파이프라인 구성
num_features = ['exclusive_area', 'price']
cat_features = ['dong']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# 3. 전체 파이프라인 구성 (preprocessor + LOF)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lof', LocalOutlierFactor(n_neighbors=20, contamination=0.03))
])

# 4. 모델 학습
X = df.copy()
pipeline.fit(X)

# 5. LOF 점수 계산 (후처리용)
X_transformed = pipeline.named_steps['preprocessor'].transform(X)
lof_model = pipeline.named_steps['lof']
lof_scores = -lof_model.negative_outlier_factor_
score_min, score_max = lof_scores.min(), lof_scores.max()

# 6. 예측 함수 (Flask에서 사용 가능)
def predict_lof_score(dong: str, exclusive_area: float, price: int):
    input_df = pd.DataFrame([[dong, exclusive_area, price]], columns=['dong', 'exclusive_area', 'price'])
    
    # 변환 및 희소 행렬 → 밀집 배열로 변환
    input_transformed = pipeline.named_steps['preprocessor'].transform(input_df).toarray()
    full_transformed = X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed

    # 거리 계산
    distances = pairwise_distances(input_transformed, full_transformed)
    neighbors_idx = np.argsort(distances[0])[:20]

    # 각 이웃에 대한 local reachability density 계산
    def get_lrd(index):
        d = pairwise_distances(full_transformed[[index]], full_transformed)
        neighbor_dists = d[0][np.argsort(d[0])[:20]]
        return 1 / np.mean(np.maximum(neighbor_dists, 1e-10))

    lrd_input = 1 / np.mean(np.maximum(distances[0][neighbors_idx], 1e-10))
    lrd_neighbors = np.mean([get_lrd(i) for i in neighbors_idx])

    lof_score = lrd_neighbors / lrd_input
    anomaly_score = (lof_score - score_min) / (score_max - score_min)

    return round(anomaly_score, 3)

# 7. 사용 예시
print(predict_lof_score("풍암동", 84.3, 65000))  # anomaly_score 출력