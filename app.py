from flask import Flask, request, jsonify
import pandas as pd
import joblib
import psycopg2

app = Flask(__name__)

# ✅ 1. 모델 로드
model = joblib.load("xgb_weighted_model.pkl")

# ✅ 2. PostgreSQL 접속 정보 (당신 환경에 맞게 비밀번호 수정)
DB_INFO = {
	'host': 'localhost',
	'port': 5433,
	'dbname': 'postgres',  # 확인된 DB 이름
	'user': 'postgres',
	'password': 'dido222!!'  # ← 설치 시 설정한 비밀번호
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

# ✅ 3. 예측 결과 DB 저장 함수
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
        "💡 사용 방법:\n1. 보증금, 월세, 전용면적, 방 수, 욕실 수, 관리비를 입력해 주세요.\n2. AI가 허위매물 여부를 예측해드립니다!"
    )

# ✅ [restart] 블록
@app.route("/restart", methods=["POST"])
def restart():
    return simple_response("새로운 매물 정보를 입력해 주세요. 다시 예측을 시작합니다!")
	
# ✅ 4. 예측 엔드포인트 (카카오 챗봇 연결)
@app.route('/predict', methods=['POST'])
def predict():
	try:
		# 입력 데이터 파싱
		data = request.get_json()
		params = data['action']['params']

		# 입력값 처리
		input_data = {
			'deposit': int(params.get('deposit', 0)),
			'monthly_rent': int(params.get('monthly_rent', 0)),
			'area': float(params.get('area', 0)),
			'rooms': int(params.get('rooms', 0)),
			'bathrooms': int(params.get('bathrooms', 0)),
			'admin_fee': int(params.get('admin_fee', 0))
		}

		
		print("📦 받은 params 전체:", params)
		print("📥 파싱한 input_data:", input_data)
		
		
		columns = ['deposit', 'monthly_rent', 'area', 'rooms', 'bathrooms', 'admin_fee']
		df = pd.DataFrame([input_data])[columns]

		print("🧾 최종 df:", df)
		

		# 예측
		prediction = int(model.predict(df)[0])
		probability = float(model.predict_proba(df)[0][1])

		# 결과 저장을 위해 dict에 추가
		input_data['prediction'] = prediction
		input_data['probability'] = probability

		# DB 저장
		insert_prediction_to_db(input_data)

		# 메시지 응답 구성
		msg = (
			f"⚠️ 허위매물일 가능성이 높습니다. (확률: {probability:.2%})"
			if prediction == 1 else
			f"✅ 정상 매물일 가능성이 높습니다. (확률: {1 - probability:.2%})"
		)

		# 카카오톡 응답 형식
		return jsonify({
			"version": "2.0",
			"template": {
				"outputs": [
					{
						"simpleText": {
							"text": msg
						}
					}
				]
			}
		})

	except Exception as e:
		print("[ERROR]", e)
		return jsonify({
			"version": "2.0",
			"template": {
				"outputs": [
					{
						"simpleText": {
							"text": "❌ 예측 중 오류가 발생했습니다. 서버 로그를 확인해주세요."
						}
					}
				]
			}
		})

# ✅ 5. 실행
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)
