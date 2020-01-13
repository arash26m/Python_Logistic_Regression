#---------#R11~S11  
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

address = 'C:/Users/ataghi2/OneDrive/Arash/Papers/James/CH4DATA to Arash2.csv'
df = pd.read_csv(address)

#df.head(10)
#print("Count: " +str(len(df.index)))
#sns.countplot(x="R11", hue="S11", data=df)
#df["PBC"].plot.hist()

X= df.loc[:,['S11']].values
y= df.loc[:,["R11"]].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X, y)
print(logmodel.intercept_)
print(logmodel.coef_)

predictions = logmodel.predict(X)

from sklearn.metrics import classification_report
classification_report(y, predictions)

from sklearn.metrics import confusion_matrix
confusion_matrix(y, predictions)

from sklearn.metrics import accuracy_score
accuracy_score(y, predictions)

#---------#R12~S12  
X= df.loc[:,['S12']].values
y= df.loc[:,["R12"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
logmodel.fit(X, y)
print(logmodel.intercept_)
print(logmodel.coef_)
predictions = logmodel.predict(X)
classification_report(y, predictions)
confusion_matrix(y, predictions)
accuracy_score(y, predictions)

#---------#R73~S73  
X= df.loc[:,['S73']].values
y= df.loc[:,["R73"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
logmodel.fit(X, y)
print(logmodel.intercept_)
print(logmodel.coef_)
predictions = logmodel.predict(X)
classification_report(y, predictions)
confusion_matrix(y, predictions)
accuracy_score(y, predictions)

#---------#PBA~RQL1J+RQL2J...+RQL7J
X= df.loc[:,['RQL1J','RQL2J','RQL3J','RQL4J','RQL5J','RQL6J','RQL7J']].values
y= df.loc[:,["PBA"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
logmodel.fit(X, y)
print(logmodel.intercept_)
print(logmodel.coef_)
predictions = logmodel.predict(X)
classification_report(y, predictions)
confusion_matrix(y, predictions)
accuracy_score(y, predictions)
