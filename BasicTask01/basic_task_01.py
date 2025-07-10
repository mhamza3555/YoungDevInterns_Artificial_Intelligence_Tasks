import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("loading the dataset")
data = pd.read_csv('student_scores.csv')

print("preparing data")
print("X is feature")
print("Y will be target")
X = data[['Hours']]
y = data['Scores'] 

print("splitting the data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

print("training the model")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("evaluating the model using MAE and R2 socre:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

print("Predicting score for 9.25 study hours")
custom_hours = pd.DataFrame([[9.25]], columns=["Hours"])
predicted_score = model.predict(custom_hours)
print(f"Predicted score for 9.25 study hours = {predicted_score[0]:.2f}")