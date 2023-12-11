import pandas as pd
df = pd.read_csv("diabetes.csv")
df

X = df.drop(["Outcome"], axis =1)
y= df["Outcome"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=50)

from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
