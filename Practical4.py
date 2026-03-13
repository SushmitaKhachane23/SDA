# Import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# A. SVM with Linear Kernel

svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)

y_pred_linear = svm_linear.predict(X_test)

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("\nLinear Confusion Matrix:\n", confusion_matrix(y_test, y_pred_linear))



# B. SVM with RBF Kernel

svm_rbf = SVC(kernel='rbf', C=1, gamma=0.1)
svm_rbf.fit(X_train, y_train)

y_pred_rbf = svm_rbf.predict(X_test)

print("\nRBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("\nRBF Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rbf))



# C. Hyperparameter Tuning

param_grid = {
'C': [0.1, 1, 10],
'gamma': [0.01, 0.1, 1]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

# Evaluate tuned model
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\nTuned RBF SVM Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nTuned Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))