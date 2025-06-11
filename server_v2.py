from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# 1) 모델 로드
with open("rent_encoder.pkl","rb") as f: rent_encoder = pickle.load(f)
with open("rent_scaler.pkl","rb") as f:  rent_scaler  = pickle.load(f)
with open("rent_lof.pkl","rb") as f:     rent_model   = pickle.load(f)
with open("sale_scaler.pkl","rb") as f:  sale_scaler  = pickle.load(f)
with open("sale_iso.pkl","rb") as f:     sale_model   = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # 기대 입력 예시:
    # {
    #   "type": "전세",
    #   "dong": "광주광역시 남구 봉선동",
    #   "area": 84.64,
    #   "deposit": 20000,
    #   "monthly_rent": 0,        # 전세면 0
    #   "price": null            # 매매면 null
    # }

    t = data.get("type")
    dong = data.get("dong")
    area = float(data.get("area", 0))
    # 전월세
    if t in ["전세", "월세"]:
        deposit = float(data.get("deposit", 0))
        monthly = float(data.get("monthly_rent", 0))
        # 2) 원-핫 인코딩
        rent_type_arr = rent_encoder.transform([[t]])
        # 3) 특성 벡터 구성
        x = np.hstack([[area, deposit, monthly], rent_type_arr[0]])
        # 4) 스케일링
        x_scaled = rent_scaler.transform([x])
        # 5) 이상도 점수: LOF novelty 모드에서는 negative_outlier_factor_ 사용
        score = -rent_model.decision_function(x_scaled)[0]
        label = int(rent_model.predict(x_scaled)[0])  # -1: 이상치, 1: 정상
    # 매매
    elif t == "매매":
        price = float(data.get("price", 0))
        x = np.array([[area, price]])
        x_scaled = sale_scaler.transform(x)
        # IsolationForest
        score = -sale_model.decision_function(x_scaled)[0]
        label = int(sale_model.predict(x_scaled)[0])
    else:
        return jsonify({"error": "type must be one of 전세, 월세, 매매"}), 400

    # 6) 응답 구성
    return jsonify({
        "dong": dong,
        "type": t,
        "area": area,
        "deposit": deposit if t!="매매" else None,
        "price": price if t=="매매" else None,
        "monthly_rent": monthly if t!="매매" else None,
        "anomaly_label": label,
        "anomaly_score": float(score),
        "message": "이 값은 정상(1) / 이상치(-1)로 분류되었으며, 이상도 점수가 높을수록 더 동떨어져 있습니다."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)