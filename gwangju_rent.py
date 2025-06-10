import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# 1. CSV 로딩 및 전처리
df_rent = pd.read_csv("gwangju_rent.csv", encoding='cp949', skiprows=15)
df_rent = df_rent[['시군구', '전월세구분', '전용면적(㎡)', '보증금(만원)', '월세금(만원)']].dropna()
df_rent['dong'] = df_rent['시군구'].apply(lambda x: str(x).split()[-1])
df_rent = df_rent.rename(columns={
    '전월세구분': 'transaction_type',
    '전용면적(㎡)': 'exclusive_area',
    '보증금(만원)': 'deposit',
    '월세금(만원)': 'monthly_rent'
})
df_rent['deposit'] = df_rent['deposit'].replace(',', '', regex=True).astype(int)
df_rent['monthly_rent'] = df_rent['monthly_rent'].astype(float)

# 2. 월세만 필터링 + 샘플링
df_rent_model = df_rent[df_rent['transaction_type'] == '월세']
df_rent_sample = df_rent_model.sample(5000, random_state=42) if len(df_rent_model) > 5000 else df_rent_model

# 3. 전처리 + 모델
rent_preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['exclusive_area', 'deposit', 'monthly_rent']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['dong'])
])
rent_pipeline = Pipeline(steps=[
    ('preprocessor', rent_preprocessor),
    ('lof', LocalOutlierFactor(n_neighbors=20, contamination=0.03))
])

# 4. 학습
X_rent = df_rent_sample[['dong', 'exclusive_area', 'deposit', 'monthly_rent']].copy()
rent_pipeline.fit(X_rent)

# 5. 점수 범위 저장
X_rent_trans = rent_pipeline.named_steps['preprocessor'].transform(X_rent)
lof_scores = -rent_pipeline.named_steps['lof'].negative_outlier_factor_
score_min, score_max = lof_scores.min(), lof_scores.max()

# 6. 저장
joblib.dump(rent_pipeline, "rent_lof_model.pkl")
joblib.dump((X_rent_trans, score_min, score_max), "rent_lof_meta.pkl")

print("✅ rent_lof_model.pkl / rent_lof_meta.pkl 저장 완료!")
