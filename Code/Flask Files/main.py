import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import pickle

df = pd.read_csv('BodyFitnessPrediction.csv')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['bool_of_active']=le.fit_transform(df['bool_of_active'])
df['mood']=le.fit_transform(df['mood'])

#Independent variables
x=df.iloc[:,[1,2,3,4,6]]
#Dependent Variable
y=df.iloc[:,5]

from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder(categorical_features=[1])
x=oh.fit_transform(x).toarray()
x=x[:,1:]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=0)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

app = Flask(__name__)
model = pickle.load(open('gbmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    a=int_features
    if(a[0]=='sad'):
        a.pop(0)
        a=[0,0]+a
    elif(a[0]=='neutral'):
        a.pop(0)
        a=[1,0]+a
    elif(a[0]=='happy'):
        a.pop(0)
        a=[0,1]+a
    
    for i in range(len(a)):
        a[i]=int(a[i])
    
    final_features = scaler.transform(np.array([a]))
    prediction = model.predict(final_features)
    if(prediction[0] == 0):
        output="Inactive"
    else:
        output="Active"

    return render_template('index.html', prediction_text='Body Fitness Level : {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    a=list(data.values())
    if(a[0]=='sad'):
        a.pop(0)
        a=[0,0]+a
    elif(a[0]=='neutral'):
        a.pop(0)
        a=[1,0]+a
    elif(a[0]=='happy'):
        a.pop(0)
        a=[0,1]+a
    
    for i in range(len(a)):
        a[i]=int(a[i])
        
    prediction = model.predict(scaler.transform(np.array([a])))
    if(prediction[0] == 0):
        output="Inactive"
    else:
        output="Active"
        
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)