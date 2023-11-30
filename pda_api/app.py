from flask import Flask, request, send_from_directory
from model import kNN
import numpy as np
from math import floor

app = Flask(__name__)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/styles.css')
def styles():
    return send_from_directory('static', 'styles.css')


@app.post('/api')
def predict():
    x = np.array(list(request.form.to_dict().values())).reshape(1, -1).astype(float)
    prediction = kNN.predict(x)
    return {"prediction": floor(prediction[0])}


if __name__ == '__main__':
    app.run()
