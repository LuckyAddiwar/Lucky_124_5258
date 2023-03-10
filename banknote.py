import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('BankNote_Authentication.csv')

print(dataset.dtypes)
print(dataset.shape)

dataset.head()

# Splitting the data into x and Y sets:
X=dataset.drop(['class'],axis=True)
Y=dataset['class']

#Logistic Regression: Sklearn
from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(X,Y)

pickle.dump(log, open('BankNote.pkl','wb'))

model = pickle.load(open('BankNote.pkl','rb'))

print(model.predict([[3.8,8.9,-0.44,-2.3]]))

