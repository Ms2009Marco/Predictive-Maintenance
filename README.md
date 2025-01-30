# Predictive-Maintenance

Predictive Maintenance Analysis aims to provide a comprehensive analysis of a predictive maintenance dataset using machine learning techniques. The primary objective of this project is to predict machine failures and classify the type of failure using Random Forest models. Additionally, SHAP (SHapley Additive exPlanations) is employed to enhance interpretability and provide insight into feature importance and model decisions.

This project includes several key features. First, it offers failure prediction, enabling the identification of whether a machine is likely to fail. For cases where failures are detected, it also classifies the type of failure that occurs. Feature engineering techniques are applied to derive new features, such as temperature differences, torque-speed interaction, and squared tool wear, which enhance the model's predictive performance. The project evaluates model performance through various metrics, including confusion matrices, ROC curves, and classification reports. Moreover, interpretability is a significant focus, as SHAP analysis is used to explore feature importance through visualizations like beeswarm plots and force plots.

The project structure is organized into three main components. The first is data preprocessing, where categorical data is encoded, new features are engineered, and all features are standardized for model input. Next is failure prediction, which involves training a Random Forest Classifier to predict failures and evaluating its performance with confusion matrices, ROC curves, feature importance plots, and SHAP analysis. Finally, failure type classification focuses on training another Random Forest Classifier to classify the type of failure for instances where failures occur. This step also includes performance evaluation using confusion matrices and classification reports.

The visualizations generated in this project include confusion matrices to visualize correct and incorrect predictions, ROC curves to evaluate the model’s ability to distinguish between classes, and feature importance plots that highlight the most influential features. SHAP plots are integral to the project, with beeswarm plots showing the global impact of features on predictions and force plots explaining individual predictions.

Data-Set from Kaggle:

https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/data
