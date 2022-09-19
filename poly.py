# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# import dataset
df = pd.read_csv('Position_Salaries.csv')
# extract independent and dependent variables
x_ax = df.iloc[:, 1:2]
y_ax = df.iloc[:, 2]
# fitting the linear regression to the dataset
# train linear regression on whole dataset
linear_regs = LinearRegression()
linear_regs.fit(x_ax, y_ax)
# fit polynomial regression to dataset
# train polynomial regression on whole dataset
poly_regs = PolynomialFeatures(degree=4)
x_poly = poly_regs.fit_transform(x_ax)
linear_reg2 = LinearRegression()
linear_reg2.fit(x_poly, y_ax)
# visualising result for linear regression model
plt.scatter(x_ax, y_ax, color="purple")
plt.plot(x_ax, linear_regs.predict(x_ax), color="cyan")
plt.title("Position Level and Salary")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
# visualising result for polynomial regression model
plt.scatter(x_ax, y_ax, color="purple")
plt.plot(x_ax, linear_reg2.predict(poly_regs.fit_transform(x_ax)), color="cyan")
plt.title("Position Level and Salary")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
predict = linear_regs.predict([[5.5]])
print(predict)
