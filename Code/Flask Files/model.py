# Importing the libraries
import numpy as np
import pandas as pd
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

#Splitting the dataset into Train set and Test set
from sklearn import model_selection, neighbors
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=0)

#Apply normalization to rescale the features to a standard range of values.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(max_depth=3)
gb.fit(x_train,y_train)

# Saving model to disk
pickle.dump(gb, open('gbmodel.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('gbmodel.pkl','rb'))

print(model.predict(scaler.transform(np.array([[1,0,5464,181,5,66]]))))