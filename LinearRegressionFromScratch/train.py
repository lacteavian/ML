import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from linearregression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=4, noise=20, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
reg = LinearRegression(lr=0.01, n_iters=100)
reg.fit(X_train, Y_train)
y_pred = reg.predict(X_test)
print(np.mean((Y_test - y_pred)**2))