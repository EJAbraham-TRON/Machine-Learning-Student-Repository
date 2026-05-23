import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB


# Load the dataset
df = pd.read_csv('Maternal Health Risk Data Set.csv')
df


# DATA CLEANING

# Check for missing values
df.isnull().sum()

# Duplicates check
print("Number of duplicates found:", df.duplicated().sum())

# Note: dropping duplicates reduces the dataset from 1,014 to 452 rows(I tried this in the first draft of the assignment)
# which compromises the minimum row requirement for this assignment.
# Duplicates are therefore retained as they may represent different patients
# with identical clinical measurements rather than data entry errors.

print("DataFrame shape:", df.shape)
# EXPLORATORY DATA ANALYSIS

# Summary statistics
summary_stats = df.describe()
print("Summary Statistics:")
summary_stats

# Class distribution
colors = ["green", "orange", "red"]
sns.countplot(x='RiskLevel', data=df, palette=colors,
              order=['low risk', 'mid risk', 'high risk'])
plt.title('Class Distribution — Maternal Health Risk')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.show()

# Feature distributions by risk level
features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.boxplot(x='RiskLevel', y=feature, data=df,
                order=['low risk', 'mid risk', 'high risk'],
                palette=colors, ax=axes[i])
    axes[i].set_title(feature, fontweight='bold')
    axes[i].set_xlabel('')

plt.suptitle('Feature Distributions by Risk Level', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# DATA PREPROCESSING

# Encode target variable
le = LabelEncoder()
df['RiskEncoded'] = le.fit_transform(df['RiskLevel'])

# Separate features and target
X = df[features]
y = df['RiskEncoded']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# StandardScaler for distance-based classifiers and Gaussian NB
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# MinMaxScaler for Bernoulli and Complement NB
mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train)
X_test_mm  = mm_scaler.transform(X_test)


# DEFINE CLASSIFIERS

classifiers = {
    "KNN (k=3)":        (KNeighborsClassifier(n_neighbors=3), X_train_scaled, X_test_scaled),
    "KNN (k=7)":        (KNeighborsClassifier(n_neighbors=7), X_train_scaled, X_test_scaled),
    "SVM (RBF Kernel)": (SVC(kernel='rbf', probability=True), X_train_scaled, X_test_scaled),
    "Gaussian NB":      (GaussianNB(),    X_train_scaled, X_test_scaled),
    "Bernoulli NB":     (BernoulliNB(),   X_train_mm,     X_test_mm),
    "Complement NB":    (ComplementNB(),  X_train_mm,     X_test_mm),
}


# TRAIN THE MODELS

results = []
trained_models = {}

for name, (clf, X_tr, X_te) in classifiers.items():
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te) if hasattr(clf, 'predict_proba') else None

    results.append({
        'Classifier': name,
        'Accuracy':   round(accuracy_score(y_test, y_pred),                                    4),
        'Precision':  round(precision_score(y_test, y_pred, average='weighted'),               4),
        'Recall':     round(recall_score(y_test, y_pred, average='weighted'),                  4),
        'F1 Score':   round(f1_score(y_test, y_pred, average='weighted'),                      4),
        'ROC AUC':    round(roc_auc_score(y_test, y_prob, multi_class='ovr',
                            average='weighted'), 4) if y_prob is not None else 'N/A'
    })
    trained_models[name] = (clf, X_te)


# EVALUATION METRICS

# Metrics table
metrics_df = pd.DataFrame(results)
print("Evaluation Metrics:")
print(metrics_df.to_string(index=False))

# Confusion matrices
class_names = le.classes_
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, (name, (clf, X_te)) in enumerate(trained_models.items()):
    y_pred = clf.predict(X_te)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=False)
    axes[idx].set_title(name, fontweight='bold')
    axes[idx].set_ylabel('True label')
    axes[idx].set_xlabel('Predicted label')
    axes[idx].tick_params(axis='x', rotation=15)

plt.suptitle('Confusion Matrices — All Classifiers', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# Metrics comparison bar chart
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(metrics_df))
width = 0.18
colors_bar = ['Blue', 'Red', 'Green', 'Orange']

plt.figure(figsize=(12, 5))
for i, metric in enumerate(metrics_to_plot):
    plt.bar(x + i * width, metrics_df[metric].astype(float),
            width, label=metric, color=colors_bar[i], alpha=0.85)

plt.xticks(x + width * 1.5, metrics_df['Classifier'], rotation=15, fontsize=9)
plt.ylim(0, 1.1)
plt.ylabel('Score')
plt.title('Classifier Performance Comparison — Maternal Health Risk')
plt.legend()
plt.tight_layout()
plt.show()

# Classification report for best model
best_name = metrics_df.loc[metrics_df['F1 Score'].astype(float).idxmax(), 'Classifier']
best_clf, best_Xte = trained_models[best_name]
y_pred_best = best_clf.predict(best_Xte)
print(f"\nBest Classifier: {best_name}")
print(classification_report(y_test, y_pred_best, target_names=class_names))