# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 22:48:58 2020

@author: sa_kh
"""

import numpy as np
from flask import Flask , request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html', prediction_text = 'Employee salay should be $ {}'.format(output))


# for API
@app.route('/predict_api', methods=['POST'])
def predict_API():
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

