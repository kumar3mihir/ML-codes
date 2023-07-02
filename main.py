import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
print(diabetes.keys())
#slicing
# diabetes_x = diabetes.data[:, np.newaxis, 2]
diabetes_x = diabetes.data
diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_x_test)

print("Mean squared error:", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

# plt.scatter(diabetes_x_test, diabetes_y_test)
# plt.plot(diabetes_x_test, diabetes_y_predicted, color='r')
# plt.show()
# Mean squared error: 3035.0601152912695
# Weights: [941.43097333]
# Intercept: 153.39713623331644

# after removing slicing
#more data more features will give you better sse
# Mean squared error: 1826.4841712795057
# Weights: [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
#   458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
# Intercept: 153.05824267739402