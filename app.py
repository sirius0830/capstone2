from flask import Flask, request, jsonify
import pandas as pd
import joblib
import psycopg2

app = Flask(__name__)

# ✅ 1. 모델 로드
model = joblib.load("xgb_weighted_model.pkl")

# ✅ 2. PostgreSQL 접속 정보
DB_INFO = {
    'host': 'localhost',
    'port': 5433,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'dido222!!'
}

def simple_response(text):
    return jsonify({
        "version": "2.0",
        "template": {
            "outputs": [
                { "simpleText": { "text": text } }
            ]
        }
    })

# ✅ 3. 예측 결과 DB 저장
def insert_prediction_to_db(data: dict):
    conn = psycopg2.connect(**DB_INFO)
    cur = conn.cursor()
    sql = """
        INSERT INTO prediction_log
        (deposit, monthly_rent, area, rooms, bathrooms, admin_fee, prediction, probability)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.execute(sql, (
        data['deposit'],
        data['monthly_rent'],
        data['area'],
        data['rooms'],
        data['bathrooms'],
        data['admin_fee'],
        data['prediction'],
        data['probability']
    ))
    conn.commit()
    cur.close()
    conn.close()

# ✅ [start] 블록
@app.route("/start", methods=["POST"])
def start():
    return simple_response("안녕하세요! 👋\nAI가 매물의 허위 여부를 예측해드립니다.\n예측을 시작해볼까요?")

# ✅ [help] 블록
@app.route("/help", methods=["POST"])
def help():
    return simple_response(
        "💡 사용 방법:\n아래와 같이 6개의 숫자를 쉼표로 구분해서 입력해주세요:\n보증금,월세,면적,방수,욕실수,관리비\n예: 2000,40,40,3,2,10"
    )

# ✅ [restart] 블록
@app.route("/restart", methods=["POST"])
def restart():
    return simple_response("새로운 매물 정보를 입력해 주세요. 다시 예측을 시작합니다!")

# ✅ [predict] 블록 – 쉼표 입력 지원
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw_input = data.get('userRequest', {}).get('utterance', '')
        print("🟢 userRequest.utterance:", raw_input)

        parts = [p.strip() for p in raw_input.split(',')]

        if len(parts) != 6:
            return simple_response("❗ 6개의 값을 정확히 쉼표로 구분해주세요.\n예: 2000,40,40,3,2,10")

        input_data = {
            'deposit': int(parts[0]),
            'monthly_rent': int(parts[1]),
            'area': float(parts[2]),
            'rooms': int(parts[3]),
            'bathrooms': int(parts[4]),
            'admin_fee': int(parts[5])
        }

        print("📥 파싱한 input_data:", input_data)

        df = pd.DataFrame([input_data])
        print("🧾 최종 df:", df)

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        input_data['prediction'] = prediction
        input_data['probability'] = probability

        insert_prediction_to_db(input_data)

        print(f"🔍 예측: {prediction}, 확률: {probability:.4f}")

        msg = (
            f"⚠️ 허위매물일 가능성이 높습니다. (확률: {probability:.2%})"
            if prediction == 1 else
            f"✅ 정상 매물일 가능성이 높습니다.\n예측 확률: {(1 - probability) * 100:.2f}%"
        )

        return simple_response(msg)

    except Exception as e:
        print("[ERROR]", e)
        return simple_response("❌ 예측 중 오류가 발생했습니다. 입력 형식을 다시 확인해주세요.")

# ✅ Flask 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
