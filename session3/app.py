from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.josn
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'prediction': prediction.lolist()})


if __name__ == 'main':
    app.run(Deubg=True, host='0.0.0.0')
