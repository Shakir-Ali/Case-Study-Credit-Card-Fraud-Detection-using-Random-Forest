import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
data = pd.read_csv('creditcard.csv')

## Data Preprocessing

from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time', 'Amount'], axis = 1)

x = data.drop(['Class'], axis = 1)
y = data['Class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

## Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(x_train, y_train.values.ravel())
y_pred = classifier.predict(x_test)

## Confusion Matrix

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred.round())
print(cm)
sns.heatmap(cm, annot = True)
plt.show()
print(classification_report(y_test, y_pred.round()))
