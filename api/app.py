from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('../airflow/dags/models/iris_ann_api.pkl', 'rb') as f:
    hidden_weights, hidden_bias, output_weights, output_bias = pickle.load(f)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(x):
    hidden_layer_activation = np.dot(x, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    output = sigmoid(output_layer_activation)
    return np.argmax(output, axis=1)


def convert_payload(payload):
    data = np.array([list(d.values()) for d in payload])
    return data


@app.route('/predict', methods=['POST'])
def api():
    payload = request.json
    data = convert_payload(payload)
    result = predict(data)
    return jsonify(result.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
