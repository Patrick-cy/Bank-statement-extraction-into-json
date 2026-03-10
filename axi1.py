import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, brier_score_loss, roc_auc_score, 
                             confusion_matrix, classification_report, RocCurveDisplay)
from sklearn.calibration import CalibrationDisplay

# --- 1. SETUP AND PATHS ---
# Update these paths to match your local file locations
MODEL_PATH_A = "/Users/patrickcyuzuzo/My_LLM/Modals/add_model_A.json"
MODEL_PATH_B = "/Users/patrickcyuzuzo/My_LLM/Modals/add_model_B.json"
DATA_PATH = "/Users/patrickcyuzuzo/My_LLM/Modals/click_prediction.csv"

features = [
    'id', 'year', 'day_of_year', 'time_of_day', 'device_type', 'location', 'age',
    'browser', 'OS', 'ad_style_category', 'part_of_add_evaluation_focus_group',
    'provided_feedback_about_add', 'visited_good_0', 'visited_good_1', 'visited_good_2',
    'visited_good_3', 'visited_good_4', 'visited_good_5', 'visited_good_6',
    'visited_good_7', 'visited_good_8', 'visited_good_9', 'purchased_good_0',
    'purchased_good_1', 'purchased_good_2', 'purchased_good_3', 'purchased_good_4',
    'purchased_good_5', 'purchased_good_6', 'purchased_good_7', 'purchased_good_8',
    'purchased_good_9'
]

# --- 2. DATA PREPARATION ---
df = pd.read_csv(DATA_PATH)
X = df[features]
y = df['clicked_on_add']

# Sampling for SHAP (speed optimization for presentation)
X_sampled = X.sample(n=min(500, len(X)), random_state=42)

# ==============================================================================
# ANALYSIS: MODEL A (The Primary Candidate)
# ==============================================================================
print(f"\n{'='*20} MODEL A ANALYSIS {'='*20}")
model_a = XGBClassifier()
model_a.load_model(MODEL_PATH_A)

probs_a = model_a.predict_proba(X)[:, 1]
preds_a = model_a.predict(X)

# Metrics
print(f"Accuracy:  {accuracy_score(y, preds_a):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y, probs_a):.4f}")
print(f"Brier:     {brier_score_loss(y, probs_a):.4f}")
print("\nClassification Report Model A:\n", classification_report(y, preds_a))

# Plot 1: Confusion Matrix Model A
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y, preds_a), annot=True, fmt='d', cmap='Blues')
plt.title("Model A: Confusion Matrix (Error Profile)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot 2: ROC Curve Model A
plt.figure(figsize=(6, 5))
ax = plt.gca()
RocCurveDisplay.from_predictions(y, probs_a, ax=ax, name="Model A")
ax.plot([0, 1], [0, 1], "--", color="gray")
plt.title("Model A: ROC Curve (Separation Power)")
plt.show()

# Plot 3: Calibration Curve Model A
plt.figure(figsize=(6, 5))
ax = plt.gca()
CalibrationDisplay.from_predictions(y, probs_a, n_bins=10, strategy="quantile", ax=ax, name="Model A")
plt.title("Model A: Calibration Curve (Probability Trust)")
plt.show()

# Plot 4: SHAP Global Feature Importance (Bar Plot)
print("Generating SHAP Bar Plot for Model A...")
explainer_a = shap.TreeExplainer(model_a)
shap_values_a = explainer_a.shap_values(X_sampled)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_a, X_sampled, plot_type="bar", show=False)
plt.title("Model A: Global Feature Importance (Bar Plot)")
plt.show()

# Plot 5: SHAP Summary Plot (Beehive)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_a, X_sampled, show=False)
plt.title("Model A: Detailed Feature Impact (Beehive)")
plt.show()

# ==============================================================================
# ANALYSIS: MODEL B (The Secondary Candidate)
# ==============================================================================
print(f"\n{'='*20} MODEL B ANALYSIS {'='*20}")
model_b = XGBClassifier()
model_b.load_model(MODEL_PATH_B)

probs_b = model_b.predict_proba(X)[:, 1]
preds_b = model_b.predict(X)

# Metrics
print(f"Accuracy:  {accuracy_score(y, preds_b):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y, probs_b):.4f}")
print(f"Brier:     {brier_score_loss(y, probs_b):.4f}")
print("\nClassification Report Model B:\n", classification_report(y, preds_b))

# Plot 6: Confusion Matrix Model B
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y, preds_b), annot=True, fmt='d', cmap='Oranges')
plt.title("Model B: Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot 7: ROC Curve Model B
plt.figure(figsize=(6, 5))
ax = plt.gca()
RocCurveDisplay.from_predictions(y, probs_b, ax=ax, name="Model B", color="orange")
ax.plot([0, 1], [0, 1], "--", color="gray")
plt.title("Model B: ROC Curve")
plt.show()

# Plot 8: Calibration Curve Model B
plt.figure(figsize=(6, 5))
ax = plt.gca()
CalibrationDisplay.from_predictions(y, probs_b, n_bins=10, strategy="quantile", ax=ax, name="Model B", color="orange")
plt.title("Model B: Calibration Curve")
plt.show()

# Plot 9: SHAP Global Importance (Bar Plot) for Model B
print("Generating SHAP Bar Plot for Model B...")
explainer_b = shap.TreeExplainer(model_b)
shap_values_b = explainer_b.shap_values(X_sampled)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_b, X_sampled, plot_type="bar", show=False)
plt.title("Model B: Global Feature Importance (Bar Plot)")
plt.show()