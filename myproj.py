from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
pf = pd.read_csv("data/wilt/training.csv")
pf[pf['class'] == 'n'] = 0
pf[pf['class'] == 'w'] = 1
target = np.array(pf["class"])
target.reshape(-1,1)

data = pf[["GLCM_pan","Mean_Green","Mean_Red","Mean_NIR","SD_pan"]]
X_train, X_test, y_train, y_test = train_test_split(target, data, test_size=0.33, random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,[y_train])
pred = clf.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(score)
