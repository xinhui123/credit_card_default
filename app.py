from flask import Flask
from flask import request, render_template
import pandas as pd

from scipy import stats
import numpy as np


import joblib
model1 = joblib.load("logistic")
model2 = joblib.load("classification_tree")
model3 = joblib.load("random_forest")
model4 = joblib.load("MLP")
model5 = joblib.load("xgboost")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        income = request.form.get('income')
        age = request.form.get('age')
        loan = request.form.get('loan')
        print(income, age, loan)
        pred1 = predict_result(income, age, loan, model1, False, 'Logistic Regression')
        pred2 = predict_result(income, age, loan, model2, False, 'Classification Tree')
        pred3 = predict_result(income, age, loan, model3, False, 'random_forest')
        pred4 = predict_result(income, age, loan, model4, True, 'MLP')
        pred5 = predict_result(income, age, loan, model5, True, 'xgboost')

        final_result = pred1 + pred2 + pred3 + pred4 + pred5
        
        return(render_template('index.html', result = final_result))
    else:
        return(render_template('index.html', result = 'ready'))

def predict_result(income, age, loan, model, should_normalise, model_name):
    if should_normalise == True:
        data = pd.read_csv('Credit Card Default II (balance).csv')
        data = data[['income', 'age', 'loan']]
        
        new_data = pd.DataFrame({"income":[income],
                            "age":[age],
                            'loan':[loan]})
        data = data.append(new_data)

        data_normalized = data.copy()

        for i in data_normalized.columns:
            data_normalized[i]=stats.zscore(data_normalized[i].astype(np.float))
        
        X = [[float(data_normalized['income'].iloc[-1]), 
              float(data_normalized['age'].iloc[-1]), 
              float(data_normalized['loan'].iloc[-1])]]
    else:
        X = [[float(income), float(age), float(loan)]]
    
    pred = model.predict(X)

    if pred == 0:
        result = 'Not Default.'
    else:
        result = 'Default.'

    s = 'The credit card default result predicted by model ' + model_name + ' is: ' +result + '\n'

    return s

    
if __name__ == '__main__':
    app.run()

