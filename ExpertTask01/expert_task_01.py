import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5, 10]
}

gb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_

gb_random_search = RandomizedSearchCV(estimator=gb, param_distributions=gb_param_grid, cv=5, n_iter=10,
                                      random_state=42, scoring='accuracy', n_jobs=-1)
gb_random_search.fit(X_train, y_train)
best_gb = gb_random_search.best_estimator_

rf_pred = best_rf.predict(X_test)
print("Random Forest Results:")
print("Best Parameters:", rf_grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

gb_pred = best_gb.predict(X_test)
print("\nGradient Boosting Results:")
print("Best Parameters:", gb_random_search.best_params_)
print("Accuracy:", accuracy_score(y_test, gb_pred))
print("Classification Report:\n", classification_report(y_test, gb_pred))
