from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

x, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(x, y)

joblib.dump(model, "model.pkl")