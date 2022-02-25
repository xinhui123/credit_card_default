from flask import Flask
app = Flask(__name__)
from flask import request, render_template
import joblib
import pandas as pd
from scipy import stats
import numpy as np

def getDefault(i):
    return "Default" if i else "No Default"

@app.route("/",methods = ["GET","POST"])
def index():
    if request.method == "POST":
        df = pd.read_csv("Credit Card Default II (balance).csv")
        dfnew = df[df["age"]>0]
        x_column = ['income', 'age', 'loan']
        age = request.form.get("age")
        loan = request.form.get("loan")
        income = request.form.get("income")
        dfnew = dfnew[['income', 'age', 'loan']]
        df2 = {'income': income, 'age': age, 'loan': loan}
        dfnew = dfnew.append(df2, ignore_index=True)
        
        model1 = joblib.load("logistic")
        model2 = joblib.load("classification_tree")
        model3 = joblib.load("random_forest")
        model4 = joblib.load("xgboost")
        model5 = joblib.load("MLP")
        
        normalised_df = dfnew.copy()
        for i in x_column:
            normalised_df[i] = stats.zscore(normalised_df[i].astype(np.float))
        lastrow = normalised_df.iloc[-1]
        pred = model1.predict([[float(lastrow[0]),float(lastrow[1]),float(lastrow[2])]])
        pred2 = model2.predict([[float(income),float(age),float(loan)]])
        pred3 = model3.predict([[float(lastrow[0]),float(lastrow[1]),float(lastrow[2])]])
        pred4 = model4.predict([[float(lastrow[0]),float(lastrow[1]),float(lastrow[2])]])
        pred5 = model5.predict([[float(lastrow[0]),float(lastrow[1]),float(lastrow[2])]])
        res = [str(getDefault(pred[0])),str(getDefault(pred2[0])),str(getDefault(pred3[0])),str(getDefault(pred4[0])),str(getDefault(pred5[0]))]
        
        return (render_template("index.html",result=res))
    else:
        return (render_template("index.html",result=[]))
  
if __name__ == "__main__":
    app.run()
