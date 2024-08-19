import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
# Charger les données
df = pd.read_csv('fraudTrain.csv')


print(df.head())


print(df.info())
print(df.describe())
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])
df = pd.get_dummies(df, columns=['gender', 'category', 'state', 'job'], drop_first=True)

features = df.drop(['is_fraud', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'dob', 'trans_num', 'unix_time', 'merchant'], axis=1)
target = df['is_fraud']

print(features.shape)
print(target.shape)
print(target.isna().sum())

df_cleaned = df.dropna(subset=['is_fraud'])
features = df_cleaned.drop(['is_fraud', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'dob', 'trans_num', 'unix_time', 'merchant'], axis=1)
target = df_cleaned['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

# Standardiser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_predictions = log_model.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, log_predictions))

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)
print("Decision Tree:")
print(classification_report(y_test, tree_predictions))

forest_model = RandomForestClassifier(random_state=42)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, forest_predictions))

# Afficher les matrices de confusion
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
predictions = [log_predictions, tree_predictions, forest_predictions]

for i, pred in enumerate(predictions):
    print(f"{models[i]} Confusion Matrix:")
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d')
    plt.title(models[i])
    plt.show()
