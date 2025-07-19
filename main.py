import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
disease_df = pd.read_csv("framingham.csv")
disease_df.drop(['education'], inplace=True, axis=1)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)

# Remove NaN / NULL values
disease_df.dropna(axis=0, inplace=True)
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())

# Features and labels
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

# Normalize the dataset
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Plotting the count of TenYearCHD values
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df, palette="BuGn_r")
plt.show()

# Plot CHD distribution
disease_df['TenYearCHD'].plot(kind='hist', bins=3, color='orange')
plt.title('Distribution of TenYearCHD')
plt.xlabel('TenYearCHD')
plt.ylabel('Frequency')
plt.show()

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_predictions = logreg.predict(X_test)

# Accuracy
print('Accuracy of the model is =', accuracy_score(y_test, y_predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, y_predictions)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.title('Confusion Matrix')
plt.show()

fig , ax = plt.subplots(figsize=(10 , 5))
sns.heatmap(disease_df.corr() ,annot=True , ax=ax , cmap= 'viridis')
plt.show()

# Classification Report
print('The details for confusion matrix are:')
print(classification_report(y_test, y_predictions))

