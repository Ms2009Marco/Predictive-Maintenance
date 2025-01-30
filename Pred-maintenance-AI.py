# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import shap
from IPython.display import display, HTML  # For inline rendering of SHAP plots and HTML content

# 1. Data Preprocessing
# Load the dataset
maintenance_data = pd.read_csv('/mnt/c/Users/marco/Desktop/AI - TH-Bingen/predictive_maintenance.csv')

# Encode the Failure Type column
label_encoder = LabelEncoder()
maintenance_data['Failure_Type_Encoded'] = label_encoder.fit_transform(maintenance_data['Failure Type'])

# Create a feature for the difference between process and air temperature
maintenance_data['Temperature_Difference'] = (
    maintenance_data['Process temperature [K]'] - maintenance_data['Air temperature [K]']
)

# Add interaction terms for Torque and Rotational Speed and square Tool Wear
maintenance_data['Torque_Speed_Interaction'] = (
    maintenance_data['Torque [Nm]'] * maintenance_data['Rotational speed [rpm]']
)
maintenance_data['Tool_Wear_Squared'] = maintenance_data['Tool wear [min]'] ** 2

# Define features and target for failure prediction
predictive_features = maintenance_data[[
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Torque_Speed_Interaction',
    'Tool_Wear_Squared',
    'Temperature_Difference'
]]
failure_target = maintenance_data['Target']

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(predictive_features)

# 2. Failure Prediction
## 2.1 Model Training
# Split data into training and test sets for failure prediction
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, failure_target, test_size=0.2, random_state=42
)

# Train a Random Forest Classifier to predict failure
failure_model = RandomForestClassifier(random_state=42)
failure_model.fit(X_train, y_train)

## 2.2 Model Evaluation
# Predict and evaluate failure prediction
failure_predictions = failure_model.predict(X_test)
print("Failure Prediction Report:")
print(classification_report(y_test, failure_predictions))
print("Failure Prediction Accuracy:", accuracy_score(y_test, failure_predictions))

# Display confusion matrix for failure prediction
ConfusionMatrixDisplay.from_predictions(y_test, failure_predictions, display_labels=['No Failure', 'Failure'])
plt.title("Confusion Matrix for Failure Prediction")
plt.show()

# Explanation: Confusion Matrix
print("""
Confusion Matrix Analysis:
- **True Positives (TP):** The model correctly predicts Failures.
- **True Negatives (TN):** The model correctly predicts No Failures.
- **False Positives (FP):** The model predicts a Failure where there was none.
- **False Negatives (FN):** The model fails to detect an actual Failure.
""")

# Plot ROC and AUC curve for failure prediction
failure_probabilities = failure_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, failure_probabilities)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Explanation: ROC Curve
print(f"""
ROC Curve Analysis:
- The AUC value is {roc_auc_value:.2f}, indicating the model's ability to distinguish between Failures and No Failures.
- A value closer to 1 indicates better performance, while 0.5 would mean random guessing.
""")

## 2.3 SHAP Analysis
# SHAP Analysis for Failure Prediction
shap.initjs()  # Initialize JavaScript for SHAP plots
shap_explainer = shap.Explainer(failure_model, X_train)
shap_values = shap_explainer(X_test, check_additivity=False)  # Apply to full test set

# Feature Importance Plot
feature_importances = failure_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(predictive_features.columns, feature_importances, color='skyblue')
plt.title("Feature Importance for Failure Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Explanation: Feature Importance
print("""
Feature Importance Analysis:
- The bar chart shows how much each feature contributes to the model's predictions.
- Features like "Torque_Speed_Interaction" and "Tool_Wear_Squared" seem to have significant influence.
""")

# Beeswarm plot for SHAP values
shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=predictive_features.columns)

# Explanation: SHAP Beeswarm Plot
print("""
SHAP Beeswarm Plot Analysis:
- This plot shows the distribution and magnitude of SHAP values for all features.
- Features with wider distributions or more outliers are more influential.
""")

# Forceplots for specific cases
scenarios = {
    "True Positive": (1, 1),
    "True Negative": (0, 0),
    "False Negative": (1, 0),
    "False Positive": (0, 1)
}

for scenario, (true_label, pred_label) in scenarios.items():
    index = next(
        i for i, (true, pred) in enumerate(zip(y_test, failure_predictions))
        if true == true_label and pred == pred_label
    )
    force_plot_instance = shap.force_plot(
        shap_explainer.expected_value[1],
        shap_values.values[index, :, 1],
        pd.DataFrame(X_test, columns=predictive_features.columns).iloc[index],
        show=False
    )
    display(HTML(f"""
    <h3>Forceplot: {scenario}</h3>
    <p><strong>True Label:</strong> {true_label}</p>
    <p><strong>Predicted Label:</strong> {pred_label}</p>
    <p>This forceplot explains why the model predicted <strong>{pred_label}</strong> for an instance that is actually labeled as <strong>{true_label}</strong>. 
    The red bars represent features pushing the prediction towards "Failure," while the blue bars represent features pulling it away.</p>
    """))
    display(force_plot_instance)

# 3. Failure Type Prediction
## 3.1 Data Preparation
# Split data for failure type prediction (only on failed instances)
failed_instances = maintenance_data[maintenance_data['Target'] == 1]
failed_features = failed_instances[[
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Torque_Speed_Interaction',
    'Tool_Wear_Squared',
    'Temperature_Difference'
]]
failed_target = failed_instances['Failure_Type_Encoded']

# Standardize features for failed instances
scaled_failed_features = scaler.transform(failed_features)
X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
    scaled_failed_features, failed_target, test_size=0.2, random_state=42
)

## 3.2 Model Training and Evaluation
# Train a Random Forest Classifier for failure type prediction
failure_type_model = RandomForestClassifier(random_state=42)
failure_type_model.fit(X_train_type, y_train_type)

# Predict and evaluate failure type prediction
type_predictions = failure_type_model.predict(X_test_type)
print("Failure Type Prediction Report:")
print(classification_report(y_test_type, type_predictions))
print("Failure Type Prediction Accuracy:", accuracy_score(y_test_type, type_predictions))

# Display confusion matrix for failure type prediction
unique_classes = sorted(set(y_test_type) | set(type_predictions))
ConfusionMatrixDisplay.from_predictions(y_test_type, type_predictions, 
                                         display_labels=unique_classes)
plt.title("Confusion Matrix for Failure Type Prediction")
plt.show()
