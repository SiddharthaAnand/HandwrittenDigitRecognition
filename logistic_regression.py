'''
Logistic Regression
'''

from sklearn.linear_model import LogisticRegression

def logistic_regression(X_train, y_train, X_test):
	clf = LogisticRegression(C = 1.0)
	clf.fit(X_train, y_train)
	return clf.predict(X_test)