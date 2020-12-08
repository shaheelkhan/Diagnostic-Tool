# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:12:32 2020

@author: shahe
"""

#Import necessaryt libraries
from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__,template_folder='templates')

@app.route('/')

@app.route("/home")
def home():
    return render_template("index.html")


@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/liver')
def liver():
    return render_template('liver.html')

def FeatureChecker(vals,cols):
    predict_array = np.array(vals).reshape(1,len(cols))
    if(cols == ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']):
        
        saved_model = pickle.load(open('heart_svm.pkl','rb'))
        result = saved_model.predict(predict_array)
        
    elif(cols == ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']):
        
        saved_model = pickle.load(open('diabetes_rf.pkl','rb'))
        result = saved_model.predict(predict_array)
        
    elif(cols == ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase','Alamine_Aminotransferase', 'Albumin', 'Albumin_and_Globulin_Ratio']):
        
        saved_model = pickle.load(open('liver_gbm.pkl','rb'))
        result = saved_model.predict(predict_array)
        
    return result[0]

@app.route('/result',methods=['POST'])
def result():
    
    if request.method == 'POST':
        
        predict_list = request.form.to_dict()
        col_list = list(predict_list.keys())
        val_list = list(predict_list.values())
        val_list = list(map(float,val_list))
        
        if col_list == ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']:
            output = FeatureChecker(val_list,col_list)
            
        elif col_list == ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']:
            output = FeatureChecker(val_list,col_list)
             
        elif col_list == ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase','Alamine_Aminotransferase', 'Albumin', 'Albumin_and_Globulin_Ratio']:
            output = FeatureChecker(val_list,col_list)
        #return (render_template("results.html",prediction="Result {}".format(output)))
        
    if(int(output == 1)):
        prediction='Based on the inputs provided, your health condition is Not Satisfactory'
    else:
        prediction='Based on the inputs provided, your health condition is Satisfactory'
        
    return(render_template("results.html",prediction=prediction))
    
if __name__ == "__main__":
    app.run(debug=True)
            
    
        
        
        
            










































































