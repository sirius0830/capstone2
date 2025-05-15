# app.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return '✅ Flask 서버가 정상적으로 작동 중입니다!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    deposit = data['action']['params'].get('deposit', '0')

    return jsonify({
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": f"입력한 보증금은 {deposit}만 원입니다."
                    }
                }
            ]
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
