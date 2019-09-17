import os
from flask import Flask, render_template, request
import numpy as np
import pickle

app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


def predict(values):
    to_predict = np.array(values).reshape(1,12)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict_proba(to_predict)
    return result[0]

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1,8)
        loaded_model = pickle.load(open("model.pkl","rb"))
        result = loaded_model.predict_proba(to_predict)

        return render_template("result.html",prediction=str(round(result[0][1]*100, 2)))

if __name__ == "__main__":
    app.run()