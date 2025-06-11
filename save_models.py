import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import pandas as pd

# (예시) 학습 데이터 로드
rent_df = pd.read_csv("cleaned_gwangju_rent.csv", encoding="utf-8-sig")
sale_df = pd.read_csv("cleaned_gwangju_not_rent.csv", encoding="utf-8-sig")

# 전월세 LOF 모델 + 파이프라인
rent_encoder = OneHotEncoder(sparse=False).fit(rent_df[["rent_type"]])
X_rent = pd.concat([
    rent_df[["area","deposit","monthly_rent"]],
    pd.DataFrame(
      rent_encoder.transform(rent_df[["rent_type"]]),
      columns=rent_encoder.get_feature_names_out(["rent_type"])
    )
], axis=1)
rent_scaler = StandardScaler().fit(X_rent)
rent_model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
rent_model.fit(rent_scaler.transform(X_rent))

# 매매 IsolationForest 모델 + 파이프라인
X_sale = sale_df[["area","price"]]
sale_scaler = StandardScaler().fit(X_sale)
sale_model = IsolationForest(contamination=0.05, random_state=42)
sale_model.fit(sale_scaler.transform(X_sale))

# pickle 저장
with open("rent_encoder.pkl","wb") as f:    pickle.dump(rent_encoder, f)
with open("rent_scaler.pkl","wb") as f:     pickle.dump(rent_scaler, f)
with open("rent_lof.pkl","wb") as f:        pickle.dump(rent_model, f)
with open("sale_scaler.pkl","wb") as f:     pickle.dump(sale_scaler, f)
with open("sale_iso.pkl","wb") as f:        pickle.dump(sale_model, f)