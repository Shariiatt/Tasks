# Tasks
# **NAME : SHARIAT MUSHAHID**
**SUPERVISED ML**
**TASK : PREDICTING STUDENT SCORE BASED ON NO. OF HOURS STUDIED USING LINEAR REGRESSION**

import pandas as pd

**Importing dataset**

#importing dataset as a dataframe
data = pd.read_csv("http://bit.ly/w-data")

#assigning columns
x= 'Hours'
y= 'Scores'

**Plotting Data**

#plotting data
import matplotlib.pyplot as plt

data.plot(x,y,style='o')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

#Assigning values to X and Y

X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values


**Splitting Values**

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)


# **APPLYING LINEAR REGRESSION MODEL**

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,Y_train)

# **EVALUATION OF MODEL**

from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error # Import performance metrics, mean_squared_error (RMSE), and r2_score (Coefficient of determination)


y_predicted = model.predict(X)
LINE =model.coef_*X+model.intercept_
# model evaluation
rmse = mean_squared_error(Y, y_predicted)
r2 = r2_score(Y, y_predicted)
abs = mean_absolute_error(Y, y_predicted)
# printing values
print('Slope:' ,model.coef_)
print('Intercept:', model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)
print('Mean squared error', abs)


**PLOTTING REGRESSION LINE:**

# plotting values

# data points
plt.scatter(X, Y, s=10)
plt.xlabel('hours')
plt.ylabel('score')

# predicted values
plt.plot(X, LINE, color='r')
plt.show()

#Predicted values
print(X_test)
y_pred = model.predict(X_test)
y_pred

 **VIEWING ACTUAL AND PREDICTED VALUES TOGETHER**

Value_table = pd.DataFrame({'Given data' : Y_test , 'Predicted values' : y_pred }) 
Value_table

# **PERCENTAGE FOR STUDENT STUDYING 9.25 HOURS**

#predicting percentage for 8 hours studying student
hour = 9.25
hours = [[hour]]
pred_score = model.predict((hours))
print("score predicted for student studying for " + str(hour)+" hours is " + str(pred_score[0])+" percent")



