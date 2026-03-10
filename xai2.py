import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (r2_score, mean_absolute_error, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)


CSV_PATH = "/Users/patrickcyuzuzo/Downloads/store_locations_test.csv"
MODEL_PATH = "/Users/patrickcyuzuzo/Downloads/store_model_A.json"

feature_columns = [
    'is_franchise', 'store_size_sqm', 'planned_employees', 'parking_spots',
    'opening_hours_per_week', 'distance_to_city_center_km', 'distance_to_highway_km',
    'distance_to_main_road_km', 'public_transport_score', 'competition_count_1km',
    'competition_count_5km', 'foot_traffic_index', 'nearby_shops_count_500m',
    'is_mall_location', 'population_density', 'median_income', 'age_distribution_index',
    'unemployment_rate', 'rent_per_sqm', 'property_tax_index', 'country_Furtavia',
    'region_Bathing-Wuerttemberg', 'region_Beerlin', 'region_Dark_Forest',
    'region_Freeburg', 'region_Shire', 'city_tier_metro', 'city_tier_rural',
    'city_tier_suburban', 'city_tier_urban', 'store_type_convenience',
    'store_type_outlet', 'store_type_specialty', 'store_type_supermarket'
]


df = pd.read_csv(CSV_PATH)
X = df[feature_columns]
y_true_reg = df["annual_profit"]


def categorize_profit(val):
    if val > 350: return 1  
    if val > 150: return 0  
    return 2                

y_true_cat = y_true_reg.apply(categorize_profit)
class_names = ["Fair", "Good", "Poor"]


X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_true_reg, test_size=0.20, random_state=12345
)
_, _, y_train_cat, y_test_cat = train_test_split(
    X, y_true_cat, test_size=0.20, random_state=12345
)

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(MODEL_PATH)


y_pred_reg = xgb_model.predict(X_test)

y_pred_cat = np.array([categorize_profit(val) for val in y_pred_reg])


print("--- MODEL ACCURACY REPORT ---")
print(f"Regression R2 Score: {r2_score(y_test_reg, y_pred_reg):.4f}")
print(f"Mean Absolute Error: ${mean_absolute_error(y_test_reg, y_pred_reg):.2f}k")
print("\n--- CLASSIFICATION METRICS ---")
print(classification_report(y_test_cat, y_pred_cat, target_names=class_names))


fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test_cat, y_pred_cat, 
                                        display_labels=class_names, 
                                        cmap='Blues', ax=ax)
plt.title("Confusion Matrix: Actual vs. Predicted Performance")
plt.show()


plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain', 
                   title="Top 10 Profit Drivers (By Gain)")
plt.show()


y_train_mimic = np.array([categorize_profit(val) for val in xgb_model.predict(X_train)])
surrogate = DecisionTreeClassifier(max_depth=3, random_state=12345)
surrogate.fit(X_train, y_train_mimic)

plt.figure(figsize=(22, 10))
plot_tree(surrogate, feature_names=feature_columns, class_names=class_names, 
          filled=True, rounded=True, fontsize=11)
plt.title("Decision Logic: How the Model Classifies Locations")
plt.show()


y_test_bin = (y_test_cat == 1).astype(int) 


fpr_xgb, tpr_xgb, _ = roc_curve(y_test_bin, y_pred_reg)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)


y_surr_probs = surrogate.predict_proba(X_test)[:, 1]
fpr_surr, tpr_surr, _ = roc_curve(y_test_bin, y_surr_probs)
roc_auc_surr = auc(fpr_surr, tpr_surr)

plt.figure(figsize=(10, 7))
plt.plot(fpr_xgb, tpr_xgb, color='teal', lw=2, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_surr, tpr_surr, color='navy', lw=2, linestyle='--', label=f'Surrogate Tree (AUC = {roc_auc_surr:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Reliability in Identifying High-Profit Stores')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()