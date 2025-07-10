from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

print("results after training logistic regression model")
print("accuracy:", accuracy_score(y_test, log_preds))
print(classification_report(y_test, log_preds, target_names=iris.target_names))


tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

print("results of decision tree")
print("accuracy:", accuracy_score(y_test, tree_preds))
print(classification_report(y_test, tree_preds, target_names=iris.target_names))
