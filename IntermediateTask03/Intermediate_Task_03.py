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

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall'
}

results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)

print("Cross validation results:")
print(f"Accuracy: {results['test_accuracy'].mean():.4f}")
print(f"Precision: {results['test_precision'].mean():.4f}")
print(f"Recall: {results['test_recall'].mean():.4f}")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test Precision:", precision_score(y_test, y_pred))
print("Test Recall:", recall_score(y_test, y_pred))
