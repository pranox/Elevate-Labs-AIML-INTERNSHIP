import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.show()

dt_limited = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_limited.fit(X_train, y_train)
y_pred_dt_limited = dt_limited.predict(X_test)
acc_dt_limited = accuracy_score(y_test, y_pred_dt_limited)

plt.figure(figsize=(20,10))
plot_tree(dt_limited, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.show()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

importances = rf.feature_importances_
indices = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=indices, y=indices.index)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance from Random Forest')
plt.show()

cv_dt = cross_val_score(dt, X, y, cv=5).mean()
cv_rf = cross_val_score(rf, X, y, cv=5).mean()

print("Decision Tree Accuracy:", acc_dt)
print("Limited Depth Decision Tree Accuracy:", acc_dt_limited)
print("Random Forest Accuracy:", acc_rf)
print("Decision Tree Cross-Validation:", cv_dt)
print("Random Forest Cross-Validation:", cv_rf)
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

