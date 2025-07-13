import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

print("using titanic dataset for this task.")
df = pd.read_csv("Titanic-Dataset.csv")

print("printing first 5 rows of dataset: ")
print(df.head())

print("number of null values: ")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())

print("too many cabins are missing so removing this one.")
df.drop('Cabin', axis=1, inplace=True)

print("only two values are missing, handling it by taking mod: ")
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("dataset after handling all the null values: ")
print(df.isnull().sum())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\n Final dataset after normalization and encoding: ")
print(df.head())

df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Data has been split:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("Training Set Target Distribution (0 = Not Survived, 1 = Survived)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()
