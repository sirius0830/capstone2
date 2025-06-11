import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

# 1) 데이터 로드
# 경로를 본인 파일 위치로 수정하세요.
rent_df = pd.read_csv("cleaned_gwangju_rent.csv", encoding="utf-8-sig")
sale_df = pd.read_csv("cleaned_gwangju_not_rent.csv", encoding="utf-8-sig")

# 2) 전월세 이상치 탐지 (LOF)
# — rent_type 원-핫 인코딩
encoder = OneHotEncoder(sparse=False)
rent_type_enc = encoder.fit_transform(rent_df[["rent_type"]])
rent_enc_df = pd.DataFrame(rent_type_enc,
                           columns=encoder.get_feature_names_out(["rent_type"]),
                           index=rent_df.index)

# — 특징 행렬 결합
X_rent = pd.concat([rent_df[["area", "deposit", "monthly_rent"]], rent_enc_df], axis=1)

# — 표준화
scaler_r = StandardScaler()
X_rent_scaled = scaler_r.fit_transform(X_rent)

# — 모델 학습 & 예측
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
rent_df["anomaly_label"] = lof.fit_predict(X_rent_scaled)         # -1: 이상치, 1: 정상치
rent_df["anomaly_score"] = -lof.negative_outlier_factor_          # 클수록 이상함

# 3) 매매 이상치 탐지 (Isolation Forest)
X_sale = sale_df[["area", "price"]]
scaler_s = StandardScaler()
X_sale_scaled = scaler_s.fit_transform(X_sale)

iso = IsolationForest(contamination=0.05, random_state=42)
sale_df["anomaly_label"] = iso.fit_predict(X_sale_scaled)         # -1: 이상치, 1: 정상치
sale_df["anomaly_score"] = -iso.decision_function(X_sale_scaled)  # 클수록 이상함

# 4) 결과 확인
print("=== Rental Outliers Sample ===")
print(rent_df[rent_df["anomaly_label"] == -1]
      .sort_values("anomaly_score", ascending=False)
      .head()[["dong", "area", "deposit", "monthly_rent", "anomaly_score"]])

print("\n=== Sale Outliers Sample ===")
print(sale_df[sale_df["anomaly_label"] == -1]
      .sort_values("anomaly_score", ascending=False)
      .head()[["dong", "area", "price", "anomaly_score"]])